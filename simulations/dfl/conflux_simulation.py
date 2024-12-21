import os
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from typing import List, Dict, Optional

from accdfl.core import NodeMembershipChange
from accdfl.core.session_settings import ConfluxSettings, LearningSettings, SessionSettings
from accdfl.core.peer_manager import PeerManager

from accdfl.conflux.round import Round
from ipv8.configuration import ConfigBuilder

from simulations.learning_simulation import LearningSimulation
from simulations.logger import SimulationLoggerAdapter


class ConfluxSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.round_completed_counts: Dict[int, int] = {}
        self.data_dir = os.path.join("data", "n_%d_%s_s%d_sf%g_lr%g_sd%ddfl" % (
            self.args.peers, self.args.dataset, self.args.sample_size,
            self.args.success_fraction, self.args.learning_rate, self.args.seed))

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        builder.add_overlay("ConfluxCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()
        participants_pks = [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes]

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            batch_size=self.args.batch_size,
            weight_decay=self.args.weight_decay,
            local_steps=self.args.local_steps,
        )

        conflux_settings = ConfluxSettings(
            sample_size=self.args.sample_size,
            ping_timeout=5,
            chunks_in_sample=self.args.chunks_in_sample,
            success_fraction=self.args.success_fraction
        )

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.args.dataset,
            learning=learning_settings,
            participants=participants_pks,
            conflux_settings=conflux_settings,
            model=self.args.model,
            alpha=self.args.alpha,
            partitioner=self.args.partitioner,
        )

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, model, i=ind: self.on_round_complete(i, round_nr, model)
            node.overlays[0].setup(self.session_settings)
            node.overlays[0].model_manager.model_trainer.logger = SimulationLoggerAdapter(node.overlays[0].model_manager.model_trainer.logger, {})

        # Inject the nodes in each community (required for the model transfers)
        for node in self.nodes:
            node.overlays[0].nodes = self.nodes

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
        peers_r1 = await active_nodes[0].overlays[0].determine_available_peers_for_sample(1, self.session_settings.conflux_settings.sample_size)
        for node in self.nodes:
            overlay = node.overlays[0]
            if overlay.my_id in peers_r1:
                self.logger.info("Activating peer %s in round 1", overlay.peer_manager.get_my_short_id())
                new_round = Round(1)
                new_round.model = overlay.model_manager.model
                overlay.round_info[1] = new_round
                overlay.train_in_round(new_round)

    async def on_round_complete(self, ind: int, round_nr: int, model):
        if round_nr not in self.round_completed_counts:
            self.round_completed_counts[round_nr] = 0
        self.round_completed_counts[round_nr] += 1

        if self.args.accuracy_logging_interval > 0 and round_nr % self.args.accuracy_logging_interval == 0:
            print("Node %d compute accuracy for round %d!" % (ind, round_nr))
            accuracy, loss = self.evaluator.evaluate_accuracy(model, device_name=self.args.accuracy_device_name)

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                group = "\"s=%d\"" % (self.args.sample_size)
                out_file.write("%s,%d,%g,%s,%f,%d,%d,%f,%f\n" % (self.args.dataset, self.args.seed, self.args.learning_rate, group, get_event_loop().time(),
                                                                 ind, round_nr, accuracy, loss))

        if self.round_completed_counts[round_nr] < self.session_settings.conflux_settings.sample_size:
            return

        self.round_completed_counts.pop(round_nr)

        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        cur_time = get_event_loop().time()
        print("Round %d completed @ t=%f - bytes up: %d, bytes down: %d" % (round_nr, cur_time, tot_up, tot_down))
