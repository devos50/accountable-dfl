import asyncio
import logging
import os
import pickle
import shutil
import time
from argparse import Namespace
from base64 import b64encode
from random import Random
from typing import Dict, List, Optional

from flwr_datasets import FederatedDataset

import numpy as np

from accdfl.core.datasets import create_dataset
from accdfl.core.model_manager import ModelManager
from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import SessionSettings
from accdfl.conflux.community import ConfluxCommunity

from ipv8.configuration import ConfigBuilder
from ipv8.taskmanager import TaskManager
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.logger import SimulationLoggerAdapter


class LearningSimulation(TaskManager):
    """
    Base class for any simulation that involves learning.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.session_settings: Optional[SessionSettings] = None
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d_%s_sd%d" % (self.args.peers, self.args.dataset, self.args.seed))
        self.dataset: Optional[FederatedDataset] = None
        self.evaluator = None
        self.logger = None
        self.model_manager: Optional[ModelManager] = None

        self.loop = DiscreteLoop()
        asyncio.set_event_loop(self.loop)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = ConfigBuilder().clear_keys().clear_overlays()

        key_str = chr(peer_id).encode() * 1000
        key_base = b"LibNaCLSK:%s" % key_str[:68]
        key_material = b64encode(key_base).decode()
        builder.add_key_from_bin("my peer", key_material, file_path=os.path.join(self.data_dir, f"ec{peer_id}.pem"))
        return builder

    async def start_ipv8_nodes(self) -> None:
        for peer_id in range(1, self.args.peers + 1):
            if peer_id % 100 == 0:
                print("Created %d peers..." % peer_id)  # The logger has not been setup at this point
            endpoint = SimulationEndpoint()
            builder = self.get_ipv8_builder(peer_id).finalize()
            instance = IPv8(builder, endpoint_override=endpoint,
                            extra_communities={
                                'ConfluxCommunity': ConfluxCommunity,
                            })
            await instance.start()

            # Set the WAN address of the peer to the address of the endpoint
            for overlay in instance.overlays:
                overlay.max_peers = -1
                overlay.my_peer.address = instance.overlays[0].endpoint.wan_address
                overlay.my_estimated_wan = instance.overlays[0].endpoint.wan_address
                overlay.cancel_pending_task("_check_tasks")  # To ignore the warning for long-running tasks
                overlay.logger = SimulationLoggerAdapter(overlay.logger, {})
                overlay.peer_manager.logger = SimulationLoggerAdapter(overlay.peer_manager.logger, {})
                overlay.bw_scheduler.logger = SimulationLoggerAdapter(overlay.peer_manager.logger, {})

            self.nodes.append(instance)

    def setup_directories(self) -> None:
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_logger(self) -> None:
        root = logging.getLogger()
        root.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        root.setLevel(getattr(logging, self.args.log_level))

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = SimulationLoggerAdapter(self.logger, {})

    def ipv8_discover_peers(self) -> None:
        peers_list = [node.overlays[0].my_peer for node in self.nodes]
        for node in self.nodes:
            node.overlays[0].peers_list = peers_list

    def apply_availability_traces(self):
        if not self.args.availability_traces:
            return

        self.logger.info("Applying availability trace file %s", self.args.availability_traces)
        with open(self.args.availability_traces, "rb") as traces_file:
            data = pickle.load(traces_file)

        rand = Random(self.args.seed)
        device_ids = rand.sample(list(data.keys()), self.args.peers)
        for ind, node in enumerate(self.nodes):
            node.overlays[0].set_traces(data[device_ids[ind]])

    def apply_fedscale_traces(self):
        self.logger.info("Applying capability trace file %s", self.args.availability_traces)
        with open(os.path.join("data", "fedscale_traces"), "rb") as traces_file:
            data = pickle.load(traces_file)

        # Filter and convert all bandwidth values to bytes/s.
        data = {
            key: {
                **value,
                "communication": int(value["communication"]) * 1000 // 8  # Convert to bytes/s
            }
            for key, value in data.items()
            if int(value["communication"]) * 1000 // 8 >= self.args.min_bandwidth  # Filter based on minimum bandwidth
        }

        rand = Random(self.args.seed)
        device_ids = rand.sample(list(data.keys()), self.args.peers)

        nodes_bws: Dict[bytes, int] = {}
        for ind, node in enumerate(self.nodes):
            node.overlays[0].model_manager.model_trainer.simulated_speed = data[device_ids[ind]]["computation"]
            bw_limit: int = int(data[device_ids[ind]]["communication"])
            node.overlays[0].bw_scheduler.bw_limit = bw_limit
            nodes_bws[node.overlays[0].my_peer.public_key.key_to_bin()] = bw_limit

        for node in self.nodes:
            node.overlays[0].other_nodes_bws = nodes_bws

    def apply_diablo_traces(self):
        # Read and process the latency matrix
        bw_means = []
        with open(os.path.join("data", "diablo.txt"), "r") as diablo_file:
            rows = diablo_file.readlines()
            for row in rows:
                values = list(map(float, row.strip().split(',')))
                mean_value = np.mean(values) * 1000 * 1000 // 8
                bw_means.append(mean_value)

        nodes_bws: Dict[bytes, int] = {}
        for ind, node in enumerate(self.nodes):
            # TODO this is rather arbitrary for now
            node.overlays[0].model_manager.model_trainer.simulated_speed = 100
            bw_limit: int = bw_means[ind % len(bw_means)]
            node.overlays[0].bw_scheduler.bw_limit = bw_limit
            nodes_bws[node.overlays[0].my_peer.public_key.key_to_bin()] = bw_limit

        for node in self.nodes:
            node.overlays[0].other_nodes_bws = nodes_bws

    def apply_compute_and_bandwidth_traces(self):
        if self.args.traces == "none":
            return
        elif self.args.traces == "fedscale":
            self.apply_fedscale_traces()
        elif self.args.traces == "diablo":
            self.apply_diablo_traces()
        else:
            raise RuntimeError("Unknown traces %s" % self.args.traces)

    def apply_traces(self):
        """
        Set the relevant traces.
        """
        self.apply_availability_traces()
        self.apply_compute_and_bandwidth_traces()

        # Log these bandwidths
        with open(os.path.join(self.data_dir, "bandwidths.csv"), "w") as out_file:
            out_file.write("bandwidth\n")
            for node in self.nodes:
                out_file.write("%d\n" % node.overlays[0].bw_scheduler.bw_limit)

        self.logger.info("Traces applied!")

    def apply_latencies(self):
        """
        If specified in the settings, add latencies between the endpoints.
        """
        if not self.args.latencies_file:
            return

        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        self.logger.info("Read latency matrix with %d sites!" % len(latencies))

        # Assign nodes to sites in a round-robin fashion and apply latencies accordingly
        for from_ind, from_node in enumerate(self.nodes):
            for to_ind, to_node in enumerate(self.nodes):
                from_site_ind = from_ind % len(latencies)
                to_site_ind = to_ind % len(latencies)
                latency_ms = int(latencies[from_site_ind][to_site_ind]) / 1000
                from_node.endpoint.latencies[to_node.endpoint.wan_address] = latency_ms

        self.logger.info("Latencies applied!")

    async def setup_simulation(self) -> None:
        self.logger.info("Setting up simulation with %d peers..." % self.args.peers)
        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("dataset,seed,learning_rate,group,time,peer,round,accuracy,loss\n")

    async def start_simulation(self) -> None:
        active_nodes: List = []
        for ind, node in enumerate(self.nodes):
            if not node.overlays[0].traces or (node.overlays[0].traces and node.overlays[0].traces["active"][0] == 0):
                node.overlays[0].start()
                active_nodes.append(node)
        self.logger.info("Started %d nodes...", len(active_nodes))

        await self.start_nodes_training(active_nodes)

        if not self.dataset:
            self.dataset = create_dataset(self.session_settings)
        self.evaluator = ModelEvaluator(self.dataset, self.session_settings)

        if self.args.profile:
            import yappi
            yappi.start(builtins=True)

        start_time = time.time()
        if self.args.duration > 0:
            await asyncio.sleep(self.args.duration)
            self.logger.info("Simulation took %f seconds" % (time.time() - start_time))
            self.on_simulation_finished()
            self.loop.stop()
        else:
            self.logger.info("Running simulation for undefined time")

    async def start_nodes_training(self, active_nodes: List) -> None:
        pass

    def on_ipv8_ready(self) -> None:
        """
        This method is called when IPv8 is started and peer discovery is finished.
        """
        pass

    def get_statistics(self) -> Dict:
        # Determine both individual and aggregate statistics.
        total_bytes_up: int = 0
        total_bytes_down: int = 0
        total_train_time: float = 0
        total_network_time: float = 0

        individual_stats = {}
        for ind, node in enumerate(self.nodes):
            bytes_up = node.overlays[0].endpoint.bytes_up
            bytes_down = node.overlays[0].endpoint.bytes_down
            train_time = node.overlays[0].model_manager.model_trainer.total_training_time
            network_time = node.overlays[0].bw_scheduler.total_time_transmitting
            individual_stats[ind] = {
                "bytes_up": bytes_up,
                "bytes_down": bytes_down,
                "train_time": train_time,
                "network_time": network_time
            }

            total_bytes_up += bytes_up
            total_bytes_down += bytes_down
            total_train_time += train_time
            total_network_time += network_time

        aggregate_stats = {
            "bytes_up": total_bytes_up,
            "bytes_down": total_bytes_down,
            "train_time": total_train_time,
            "network_time": total_network_time
        }

        return {
            "time": asyncio.get_event_loop().time(),
            "global": aggregate_stats
        }

    def on_simulation_finished(self) -> None:
        if self.args.profile:
            import yappi
            yappi.stop()
            yappi_stats = yappi.get_func_stats()
            yappi_stats.sort("tsub")
            yappi_stats.save(os.path.join(self.data_dir, "yappi.stats"), type='callgrind')

    async def run(self) -> None:
        self.setup_directories()
        await self.start_ipv8_nodes()
        self.setup_logger()
        self.ipv8_discover_peers()
        self.apply_latencies()
        self.on_ipv8_ready()
        await self.setup_simulation()
        self.apply_traces()
        await self.start_simulation()
        self.on_simulation_finished()
