import os
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from random import Random
from typing import List, Dict, Optional

import torch

from accdfl.core import NodeMembershipChange
from accdfl.core.session_settings import DFLSettings, LearningSettings, SessionSettings
from accdfl.core.peer_manager import PeerManager

from accdfl.dfl.round import Round
from ipv8.configuration import ConfigBuilder

from simulations.learning_simulation import LearningSimulation
from simulations.logger import SimulationLoggerAdapter


class DFLSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.latest_accuracy_check_round: int = 0
        self.last_round_complete_time: Optional[float] = None
        self.round_durations: List[float] = []
        self.round_completed_counts: Dict[int, int] = {}
        self.best_accuracy: float = 0.0
        self.data_dir = os.path.join("data", "n_%d_%s_s%d_a%d_sf%g_lr%g_sd%ddfl" % (
            self.args.peers, self.args.dataset, self.args.sample_size, self.args.num_aggregators,
            self.args.success_fraction, self.args.learning_rate, self.args.seed))
        
        self.aggregated_models: List = []

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.args.bypass_model_transfers:
            builder.add_overlay("DFLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("DFLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        if self.args.active_participants:
            self.logger.info("Initial active participants: %s", self.args.active_participants)
            start_ind, end_ind = self.args.active_participants.split("-")
            start_ind, end_ind = int(start_ind), int(end_ind)
            participants_pks = [hexlify(self.nodes[ind].overlays[0].my_peer.public_key.key_to_bin()).decode()
                            for ind in range(start_ind, end_ind)]
            participants_ids = list(range(start_ind, end_ind))
        else:
            participants_pks = [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes]
            participants_ids = list(range(len(self.nodes)))

        # Determine who will be the aggregator
        peer_pk = None
        lowest_latency_peer_id = -1
        if self.args.fix_aggregator:
            lowest_latency_peer_id = self.determine_peer_with_lowest_median_latency(participants_ids)
            peer_pk = self.nodes[lowest_latency_peer_id].overlays[0].my_peer.public_key.key_to_bin()

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            batch_size=self.args.batch_size,
            weight_decay=self.args.weight_decay,
            local_steps=self.args.local_steps,
        )

        dfl_settings = DFLSettings(
            sample_size=self.args.sample_size,
            num_aggregators=self.args.num_aggregators,
            success_fraction=self.args.success_fraction,
            liveness_success_fraction=self.args.liveness_success_fraction,
            ping_timeout=5,
            inactivity_threshold=1000,
            fixed_aggregator=peer_pk if self.args.fix_aggregator else None,
            aggregation_timeout=self.args.aggregation_timeout,
        )

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.args.dataset,
            learning=learning_settings,
            participants=participants_pks,
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=len(self.nodes),
            dataset_base_path=self.args.dataset_base_path,
            dfl=dfl_settings,
            model=self.args.model,
            alpha=self.args.alpha,
            partitioner=self.args.partitioner,
            eva_block_size=1000,
            is_simulation=True,
            train_device_name=self.args.train_device_name,
            bypass_training=self.args.bypass_training,
        )

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, model, i=ind: self.on_round_complete(i, round_nr, model)
            node.overlays[0].setup(self.session_settings)
            node.overlays[0].model_manager.model_trainer.logger = SimulationLoggerAdapter(node.overlays[0].model_manager.model_trainer.logger, {})

        # If we fix the aggregator, we assume unlimited upload/download slots
        if self.args.fix_aggregator:
            self.logger.info("Overriding max. EVA transfers/bw limits for peer %d", lowest_latency_peer_id)
            self.nodes[lowest_latency_peer_id].overlays[0].eva.settings.max_simultaneous_transfers = 100000

        if self.args.bypass_model_transfers:
            # Inject the nodes in each community
            for node in self.nodes:
                node.overlays[0].nodes = self.nodes

        # Generated the statistics files
        with open(os.path.join(self.data_dir, "view_histories.csv"), "w") as out_file:
            out_file.write("peer,update_time,peers\n")

        with open(os.path.join(self.data_dir, "determine_sample_durations.csv"), "w") as out_file:
            out_file.write("peer,start_time,end_time\n")

        with open(os.path.join(self.data_dir, "round_durations.csv"), "w") as out_file:
            out_file.write("time\n")

        with open(os.path.join(self.data_dir, "derived_samples.csv"), "w") as out_file:
            out_file.write("peer,sample_id,sample\n")

        with open(os.path.join(self.data_dir, "events.csv"), "w") as out_file:
            out_file.write("time,peer,round,event\n")

        # Start the liveness check (every 5 minutes)
        #self.register_task("check_liveness", self.check_liveness, interval=600)

    def check_liveness(self):
        # Condition 1: At least one online node is training their model
        one_node_training: bool = False
        for node in self.nodes:
            if node.overlays[0].is_active and node.overlays[0].model_manager.model_trainer.is_training:
                one_node_training = True
                break

        # Condition 2: There is an ongoing model transfer
        one_node_sending: bool = False
        for node in self.nodes:
            if node.overlays[0].is_active and node.overlays[0].bw_scheduler.outgoing_transfers:
                one_node_sending = True
                break

        one_node_aggregating: bool = False
        for node in self.nodes:
            if node.overlays[0].is_active and node.overlays[0].aggregations:
                one_node_aggregating = True
                break

        if not one_node_training and not one_node_sending and not one_node_aggregating:
            self.flush_statistics()
            raise RuntimeError("Liveness violated - MoDeST not making progress anymore")

    async def start_nodes_training(self, active_nodes: List) -> None:
        # Update the membership status of inactive peers in all peer managers. This assumption should be
        # reasonable as availability at the very start of the training process can easily be synchronized using an
        # out-of-band mechanism (e.g., published on a website).
        active_nodes_pks = [node.overlays[0].my_peer.public_key.key_to_bin() for node in active_nodes]
        for node in self.nodes:
            peer_manager: PeerManager = node.overlays[0].peer_manager
            for peer_pk in peer_manager.last_active:
                if peer_pk not in active_nodes_pks:
                    # Toggle the status to inactive as this peer is not active from the beginning
                    peer_info = peer_manager.last_active[peer_pk]
                    peer_manager.last_active[peer_pk] = (peer_info[0], (0, NodeMembershipChange.LEAVE))

        # We will now start round 1. The nodes that participate in the first round are always selected from the pool of
        # active peers. If we use our sampling function, training might not start at all if many offline nodes
        # are selected for the first round.

        # rand_sampler = Random(self.args.seed)
        # activated_nodes = rand_sampler.sample(active_nodes, min(len(active_nodes), self.args.sample_size))
        peers_r1 = await active_nodes[0].overlays[0].determine_available_peers_for_sample(1, self.session_settings.dfl.sample_size)
        for node in self.nodes:
            overlay = node.overlays[0]
            if overlay.my_id in peers_r1:
                self.logger.info("Activating peer %s in round 1", overlay.peer_manager.get_my_short_id())
                new_round = Round(1)
                new_round.model = overlay.model_manager.model
                overlay.round_info[1] = new_round
                overlay.train_in_round(new_round)

    def compute_model_difference_score(self, models):
        """
        Computes a difference score for a list of PyTorch models.
        
        Args:
            models (list): A list of PyTorch models to compare.
        
        Returns:
            float: The difference score (0 means models are identical).
        """
        if len(models) < 2:
            raise ValueError("At least two models are required for comparison.")
        
        # Initialize a score
        difference_score = 0.0
        
        # Compare the parameters pairwise
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                for p1, p2 in zip(models[i].parameters(), models[j].parameters()):
                    difference_score += torch.sum(torch.abs(p1 - p2)).item()
        
        return difference_score

    async def on_round_complete(self, ind: int, round_nr: int, model):
        self.aggregated_models.append(model)
        if round_nr not in self.round_completed_counts:
            self.round_completed_counts[round_nr] = 0
        self.round_completed_counts[round_nr] += 1
        if self.round_completed_counts[round_nr] < self.session_settings.dfl.sample_size:
            return
        
        self.round_completed_counts.pop(round_nr)

        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        cur_time = get_event_loop().time()
        print("Round %d completed @ t=%f - bytes up: %d, bytes down: %d" % (round_nr, cur_time, tot_up, tot_down))

        if round_nr > self.latest_accuracy_check_round:
            if not self.last_round_complete_time:
                self.round_durations.append(cur_time)
            else:
                self.round_durations.append(cur_time - self.last_round_complete_time)
            self.last_round_complete_time = cur_time

        if self.args.accuracy_logging_interval > 0 and round_nr % self.args.accuracy_logging_interval == 0 and \
                round_nr > self.latest_accuracy_check_round:

            print("Will compute accuracy for round %d!" % round_nr)
            if not self.args.bypass_training:
                accuracy, loss = self.evaluator.evaluate_accuracy(model, device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                group = "\"s=%d, a=%d\"" % (self.args.sample_size, self.args.num_aggregators)
                out_file.write("%s,%d,%g,%s,%f,%d,%d,%f,%f\n" % (self.args.dataset, self.args.seed, self.args.learning_rate, group, get_event_loop().time(),
                                                                 ind, round_nr, accuracy, loss))

                if not self.args.bypass_training and self.args.store_best_models and accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    torch.save(model.state_dict(), os.path.join(self.data_dir, "best.model"))

            self.latest_accuracy_check_round = round_nr

        if self.args.rounds and round_nr >= self.args.rounds:
            print(self.compute_model_difference_score(self.aggregated_models))


            self.on_simulation_finished()
            self.loop.stop()

    def get_statistics(self) -> Dict:
        statistics = super().get_statistics()

        # Add the BW statistics of individual nodes
        def merge_dicts_sum(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1:
                    if isinstance(value, dict):
                        merge_dicts_sum(dict1[key], value)
                    else:
                        dict1[key] += value
                else:
                    if isinstance(value, dict):
                        dict1[key] = value.copy()
                        merge_dicts_sum(dict1[key], value)
                    else:
                        dict1[key] = value

        agg_bw_in_dict = {}
        agg_bw_out_dict = {}
        for ind, node in enumerate(self.nodes):
            merge_dicts_sum(agg_bw_in_dict, node.overlays[0].bw_in_stats)
            merge_dicts_sum(agg_bw_out_dict, node.overlays[0].bw_out_stats)

        statistics["global"]["bw_in"] = agg_bw_in_dict
        statistics["global"]["bw_out"] = agg_bw_out_dict

        return statistics

    def flush_statistics(self):
        """
        Flush all the statistics generated by nodes.
        """
        super().flush_statistics()

        # Write away the view histories
        with open(os.path.join(self.data_dir, "view_histories.csv"), "a") as out_file:
            for peer_id, node in enumerate(self.nodes):
                for update_time, active_peers in node.overlays[0].active_peers_history:
                    if self.args.write_view_histories:
                        active_peers_str = "-".join(active_peers)
                        out_file.write("%d,%f,%s\n" % (peer_id + 1, update_time, active_peers_str))
                node.overlays[0].active_peers_history = []

        # Write the determine sample durations
        with open(os.path.join(self.data_dir, "determine_sample_durations.csv"), "a") as out_file:
            for peer_id, node in enumerate(self.nodes):
                for start_time, end_time in node.overlays[0].determine_sample_durations:
                    out_file.write("%d,%f,%f\n" % (peer_id + 1, start_time, end_time))
                node.overlays[0].determine_sample_durations = []

        # Write away the round durations
        with open(os.path.join(self.data_dir, "round_durations.csv"), "a") as out_file:
            for round_duration in self.round_durations:
                out_file.write("%f\n" % round_duration)
            self.round_durations = []

        # Write away the derived samples
        with open(os.path.join(self.data_dir, "derived_samples.csv"), "a") as out_file:
            for peer_id, node in enumerate(self.nodes):
                for sample_id, sample in node.overlays[0].derived_samples:
                    sample_str = "-".join(sorted(sample))
                    out_file.write("%d,%d,%s\n" % (peer_id + 1, sample_id, sample_str))
                node.overlays[0].derived_samples = []

        # Write away all events
        new_events = []
        for node in self.nodes:
            for event in node.overlays[0].events:
                new_events.append(event)
            node.overlays[0].events = []

        if self.args.log_events:
            new_events = sorted(new_events, key=lambda x: x[0])
            with open(os.path.join(self.data_dir, "events.csv"), "a") as out_file:
                for event in new_events:
                    out_file.write("%f,%s,%d,%s\n" % event)
