import asyncio
import copy
import json
import pickle
import random
import time
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from math import floor
from typing import Dict, Optional, List, Tuple

import torch
from torch import nn

from ipv8.lazy_community import lazy_wrapper_wd
from ipv8.messaging.payload_headers import BinMemberAuthenticationPayload, GlobalTimeDistributionPayload
from ipv8.types import Peer
from ipv8.util import succeed

from accdfl.core import NodeMembershipChange
from accdfl.core.community import LearningCommunity
from accdfl.core.models import serialize_model, unserialize_model
from accdfl.core.session_settings import SessionSettings
from accdfl.dfl.caches import PingPeersRequestCache, PingRequestCache
from accdfl.dfl.payloads import AdvertiseMembership, PingPayload, PongPayload
from accdfl.dfl.sample_manager import SampleManager
from accdfl.util.eva.result import TransferResult


class DFLCommunity(LearningCommunity):
    community_id = unhexlify('d5889074c1e4c60423cee6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Statistics
        self.active_peers_history = []
        self.bw_in_stats: Dict[str, Dict[str, int]] = {
            "bytes": {
                "model": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
            },
            "num": {
                "model": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
            }
        }

        self.bw_out_stats: Dict[str, Dict[str, int]] = {
            "bytes": {
                "model": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
            },
            "num": {
                "model": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
            }
        }
        self.determine_sample_durations = []
        self.derived_samples: List[Tuple[int, List[str]]] = []
        self.events: List[Tuple[float, str, int, str]] = []

        # State
        self.ongoing_training_task_name: Optional[str] = None
        self.train_sample_estimate: int = 0
        self.aggregate_sample_estimate: int = 0
        self.advertise_index: int = 1
        self.aggregate_start_time = 0
        self.is_aggregating: bool = False
        self.completed_aggregation = False
        self.completed_training = False

        # Components
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup

        self.add_message_handler(AdvertiseMembership, self.on_membership_advertisement)
        self.add_message_handler(PingPayload, self.on_ping)
        self.add_message_handler(PongPayload, self.on_pong)

    def log_event(self, round: int, event: str):
        cur_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        self.events.append((cur_time, self.peer_manager.get_my_short_id(), round, event))

    def start(self, advertise_join: bool = False):
        """
        Start to participate in the training process.
        """
        super().start()

        if advertise_join:
            self.advertise_membership(NodeMembershipChange.JOIN)

    def setup(self, settings: SessionSettings):
        self.logger.info("Setting up experiment with %d initial participants and sample size %d (I am participant %s)" %
                         (len(settings.participants), settings.dfl.sample_size, self.peer_manager.get_my_short_id()))
        super().setup(settings)
        self.peer_manager.inactivity_threshold = settings.dfl.inactivity_threshold
        self.sample_manager = SampleManager(self.peer_manager, settings.dfl.sample_size, settings.dfl.num_aggregators)
        self.update_population_view_history()

    def get_round_estimate(self) -> int:
        """
        Get the highest round estimation, based on our local estimations and the estimations in the population view.
        """
        max_round_in_population_view = self.peer_manager.get_highest_round_in_population_view()
        return max(self.train_sample_estimate, self.aggregate_sample_estimate, max_round_in_population_view)

    def go_online(self):
        super().go_online()
        self.advertise_membership(NodeMembershipChange.JOIN)

    def go_offline(self, graceful: bool = True) -> None:
        super().go_offline()

        if self.is_aggregating:
            self.logger.warning("Aggregator %s went offline - this might impact liveness",
                                self.peer_manager.get_my_short_id())

        if graceful:
            self.advertise_membership(NodeMembershipChange.LEAVE)
        else:
            self.cancel_all_pending_tasks()

    def update_population_view_history(self):
        active_peers = self.peer_manager.get_active_peers()
        active_peers = [self.peer_manager.get_short_id(peer_pk) for peer_pk in active_peers]

        if not self.active_peers_history or (self.active_peers_history[-1][1] != active_peers):  # It's the first entry or it has changed
            self.active_peers_history.append((time.time(), active_peers))

    def advertise_membership(self, change: NodeMembershipChange):
        """
        Advertise your (new) membership to random peers.
        """
        self.logger.debug("Participant %s advertising its membership change to active participants",
                          self.peer_manager.get_my_short_id())

        active_peer_pks = self.peer_manager.get_active_peers()
        if self.my_id in active_peer_pks:
            active_peer_pks.remove(self.my_id)

        random_peer_pks = random.sample(active_peer_pks, min(self.sample_manager.sample_size, len(active_peer_pks)))
        for peer_pk in random_peer_pks:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Cannot find Peer object for participant %s!",
                                    self.peer_manager.get_short_id(peer_pk))
            self.logger.debug("Participant %s advertising its membership change to participant %s",
                              self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))
            global_time = self.claim_global_time()
            auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
            payload = AdvertiseMembership(self.get_round_estimate(), self.advertise_index, change.value)
            dist = GlobalTimeDistributionPayload(global_time)
            packet = self._ez_pack(self._prefix, AdvertiseMembership.msg_id, [auth, dist, payload])
            self.bw_out_stats["bytes"]["membership"] += len(packet)
            self.bw_out_stats["num"]["membership"] += 1
            self.endpoint.send(peer.address, packet)

        # Update your own population view
        info = self.peer_manager.last_active[self.my_id]
        self.peer_manager.last_active[self.my_id] = (info[0], (self.advertise_index, change))
        self.advertise_index += 1

    @lazy_wrapper_wd(GlobalTimeDistributionPayload, AdvertiseMembership)
    def on_membership_advertisement(self, peer, dist, payload, raw_data: bytes):
        """
        We received a membership advertisement from a new peer.
        """
        if not self.is_active:
            return

        self.bw_in_stats["bytes"]["membership"] += len(raw_data)
        self.bw_in_stats["num"]["membership"] += 1

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        self.logger.debug("Participant %s updating membership of participant %s",
                          self.peer_manager.get_my_short_id(), peer_id)

        change: NodeMembershipChange = NodeMembershipChange(payload.change)
        latest_round = self.get_round_estimate()
        if change == NodeMembershipChange.JOIN:
            # Do not apply this immediately since we do not want the newly joined node to be part of the next sample just yet.
            self.peer_manager.last_active_pending[peer_pk] = (
            max(payload.round, latest_round), (payload.index, NodeMembershipChange.JOIN))
        else:
            self.peer_manager.last_active[peer_pk] = (
            max(payload.round, latest_round), (payload.index, NodeMembershipChange.LEAVE))

    def determine_available_peers_for_sample(self, sample: int, count: int,
                                             getting_aggregators: bool = False) -> Future:
        if getting_aggregators and self.settings.dfl.fixed_aggregator:
            candidate_peers = [self.settings.dfl.fixed_aggregator]
        else:
            candidate_peers = self.sample_manager.get_ordered_sample_list(
                sample, self.peer_manager.get_active_peers(sample))
        self.logger.info("Participant %s starts to determine %d available peers in sample %d (candidates: %d)",
                         self.peer_manager.get_my_short_id(), count, sample,
                         len(candidate_peers))
        cache = PingPeersRequestCache(self, candidate_peers, count, sample)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def ping_peer(self, ping_all_id: int, peer_pk: bytes) -> Future:
        self.logger.debug("Participant %s pinging participant %s",
                          self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))
        peer_short_id = self.peer_manager.get_short_id(peer_pk)
        peer = self.get_peer_by_pk(peer_pk)
        if not peer:
            self.logger.warning("Wanted to ping participant %s but cannot find Peer object!", peer_short_id)
            return succeed((peer_pk, False))

        cache = PingRequestCache(self, ping_all_id, peer, self.settings.dfl.ping_timeout)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def send_ping(self, peer: Peer, identifier: int) -> None:
        """
        Send a ping message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PingPayload(self.get_round_estimate(), identifier)

        packet = self._ez_pack(self._prefix, PingPayload.msg_id, [auth, payload])
        self.bw_out_stats["bytes"]["ping"] += len(packet)
        self.bw_out_stats["num"]["ping"] += 1
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(PingPayload)
    def on_ping(self, peer: Peer, payload: PingPayload, raw_data: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.debug("Participant %s ignoring ping message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.bw_in_stats["bytes"]["ping"] += len(raw_data)
        self.bw_in_stats["num"]["ping"] += 1

        if peer_pk in self.peer_manager.last_active:
            self.peer_manager.update_peer_activity(peer_pk, max(self.get_round_estimate(), payload.round))

        self.send_pong(peer, payload.identifier)

    def send_pong(self, peer: Peer, identifier: int) -> None:
        """
        Send a pong message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PongPayload(self.get_round_estimate(), identifier)

        packet = self._ez_pack(self._prefix, PongPayload.msg_id, [auth, payload])
        self.bw_out_stats["bytes"]["pong"] += len(packet)
        self.bw_out_stats["num"]["pong"] += 1
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(PongPayload)
    def on_pong(self, peer: Peer, payload: PongPayload, raw_data: bytes) -> None:
        my_peer_id = self.peer_manager.get_my_short_id()
        peer_short_id = self.peer_manager.get_short_id(peer.public_key.key_to_bin())

        if not self.is_active:
            self.logger.debug("Participant %s ignoring ping message from %s due to inactivity",
                              my_peer_id, peer_short_id)
            return

        self.logger.debug("Participant %s receiving pong message from participant %s", my_peer_id, peer_short_id)

        self.bw_in_stats["bytes"]["pong"] += len(raw_data)
        self.bw_in_stats["num"]["pong"] += 1

        if not self.request_cache.has("ping-%s" % peer_short_id, payload.identifier):
            self.logger.warning("ping cache with id %s not found", payload.identifier)
            return

        self.peer_manager.update_peer_activity(peer.public_key.key_to_bin(),
                                               max(self.get_round_estimate(), payload.round))

        cache = self.request_cache.pop("ping-%s" % peer_short_id, payload.identifier)
        cache.on_pong()

    def train_in_round(self, round):
        self.ongoing_training_task_name = "round_%d" % round
        if not self.is_pending_task_active(self.ongoing_training_task_name):
            task = self.register_task(self.ongoing_training_task_name, self.train_in_round_coroutine, round)
            task.add_done_callback(lambda f, r=round: self.on_train_completed(f, r))

    def on_train_completed(self, _, round):
        if not self.is_active:
            return

        self.completed_training = True
        self.ongoing_training_task_name = None
        self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round)
        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(round))

    async def train_in_round_coroutine(self, round):
        """
        Participate in a round.
        """
        if round < 1:
            raise RuntimeError("Round number %d invalid!" % round)

        self.logger.info("Participant %s starts participating in round %d", self.peer_manager.get_my_short_id(), round)
        self.completed_training = False
        self.log_event(round, "start_train")

        # 1. Train the model
        await self.model_manager.train()

        self.log_event(round, "done_train")

        # It might be that we went offline at this point - check for it
        if not self.is_active:
            self.logger.warning("Participant %s went offline during model training in round %d - not proceeding", self.peer_manager.get_my_short_id(), round)
            return

        # 2. Determine the aggregators of the next sample that are available
        aggregators = await self.determine_available_peers_for_sample(round + 1, self.settings.dfl.num_aggregators,
                                                                      getting_aggregators=True)
        aggregator_ids: List[str] = [self.peer_manager.get_short_id(peer_id) for peer_id in aggregators]
        self.derived_samples.append((round + 1, aggregator_ids))
        self.logger.info("Participant %s determined %d available aggregators in sample %d: %s",
                         self.peer_manager.get_my_short_id(), len(aggregator_ids), round + 1, aggregator_ids)

        # 3. Send the trained model to the aggregators in the next sample
        task_name = "send_trained_model_%s" % round
        self.register_task(task_name, self.send_trained_model_to_aggregators, aggregators, round + 1)

    async def send_aggregated_model_to_participants(self, participants: List[bytes], model: nn.Module, sample_index: int) -> None:
        if not self.is_active:
            self.logger.warning("Participant %s not sending aggregated model due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        self.logger.info("Participant %s sending aggregated model of round %d to participants",
                         self.peer_manager.get_my_short_id(), sample_index - 1)

        # For load balancing purposes, shuffle this list
        random.shuffle(participants)

        population_view = copy.deepcopy(self.peer_manager.last_active)
        for peer_pk in participants:
            if peer_pk == self.my_id:
                model_cpy = copy.deepcopy(model)
                asyncio.get_event_loop().call_soon(self.received_aggregated_model, self.my_peer, sample_index, model_cpy)
                continue

            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                continue
            ensure_future(self.eva_send_model(sample_index, model, "aggregated_model", population_view, peer))

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()
        self.update_population_view_history()

    async def send_trained_model_to_aggregators(self, aggregators: List[bytes], sample_index: int) -> None:
        """
        Send the current model to the aggregators in a particular sample.
        """
        if not self.is_active:
            self.logger.warning("Participant %s not sending trained model due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        aggregator_ids = [self.peer_manager.get_short_id(aggregator) for aggregator in aggregators]
        self.logger.info("Participant %s sending trained model of round %d to %d aggregators in sample %d: %s",
                         self.peer_manager.get_my_short_id(), sample_index - 1, len(aggregators), sample_index, aggregator_ids)
        population_view = copy.deepcopy(self.peer_manager.last_active)

        # For load balancing purposes, shuffle this list
        random.shuffle(aggregators)

        for aggregator in aggregators:
            if aggregator == self.my_id:
                self.logger.info("Participant %s sending trained model to self", self.peer_manager.get_my_short_id())

                # Transfer the model back to the CPU
                device = torch.device("cpu")
                self.model_manager.model = self.model_manager.model.to(device)

                # Even when sending the model to oneself, serialize and deserialize the model to make sure all tensors are detached
                detached_model = unserialize_model(serialize_model(self.model_manager.model),
                                                   self.settings.dataset, architecture=self.settings.model)

                ensure_future(self.received_trained_model(self.my_peer, sample_index, detached_model))
                continue

            peer = self.get_peer_by_pk(aggregator)
            if not peer:
                self.logger.warning("Could not find aggregator peer with public key %s", hexlify(aggregator).decode())
                continue
            ensure_future(self.eva_send_model(sample_index, self.model_manager.model, "trained_model", population_view, peer))

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()

    def eva_send_model(self, round, model, type, population_view, peer):
        start_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        serialized_model = serialize_model(model)
        serialized_population_view = pickle.dumps(population_view)
        self.bw_out_stats["bytes"]["model"] += len(serialized_model)
        self.bw_out_stats["bytes"]["view"] += len(serialized_population_view)
        self.bw_out_stats["num"]["model"] += 1
        self.bw_out_stats["num"]["view"] += 1
        binary_data = serialized_model + serialized_population_view
        response = {"round": round, "type": type, "model_data_len": len(serialized_model)}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_model(peer, serialized_response, binary_data, start_time)

    def cancel_current_training_task(self):
        if self.ongoing_training_task_name and self.is_pending_task_active(self.ongoing_training_task_name):
            self.logger.info("Participant %s interrupting training task %s",
                             self.peer_manager.get_my_short_id(), self.ongoing_training_task_name)
            self.cancel_pending_task(self.ongoing_training_task_name)
            self.ongoing_training_task_name = None

    async def on_receive(self, result: TransferResult):
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.debug("Participant %s ignoring message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')
        json_data = json.loads(result.info.decode())
        serialized_model = result.data[:json_data["model_data_len"]]
        serialized_population_view = result.data[json_data["model_data_len"]:]
        received_population_view = pickle.loads(serialized_population_view)
        self.bw_in_stats["bytes"]["model"] += len(serialized_model)
        self.bw_in_stats["bytes"]["view"] += len(serialized_population_view)
        self.bw_in_stats["num"]["model"] += 1
        self.bw_in_stats["num"]["view"] += 1
        self.peer_manager.merge_population_views(received_population_view)
        self.peer_manager.update_peer_activity(result.peer.public_key.key_to_bin(),
                                               max(json_data["round"], self.get_round_estimate()))
        self.update_population_view_history()
        incoming_model = unserialize_model(serialized_model, self.settings.dataset, architecture=self.settings.model)

        if json_data["type"] == "trained_model":
            self.log_event(json_data["round"], "received_trained_model")
            await self.received_trained_model(result.peer, json_data["round"], incoming_model)
        elif json_data["type"] == "aggregated_model":
            self.log_event(json_data["round"], "received_aggregated_model")
            self.received_aggregated_model(result.peer, json_data["round"], incoming_model)

    def has_enough_trained_models(self) -> bool:
        return len(self.model_manager.incoming_trained_models) >= \
               floor(self.settings.dfl.sample_size * self.settings.dfl.success_fraction)

    def has_enough_trained_models_for_liveness(self) -> bool:
        return len(self.model_manager.incoming_trained_models) >= \
               floor(self.settings.dfl.sample_size * self.settings.dfl.liveness_success_fraction)

    async def received_trained_model(self, peer: Peer, index: int, model: nn.Module) -> None:
        model_round = index - 1  # The round associated with this model is one smaller than the sample index
        if self.shutting_down:
            self.logger.warning("Participant %s ignoring incoming trained model due to shutdown",
                                self.peer_manager.get_my_short_id())
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received trained model for round %d from participant %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        if index > self.aggregate_sample_estimate:
            self.logger.info("Participant %s received trained model for round %d for the first time - "
                             "starting to aggregate", self.peer_manager.get_my_short_id(), model_round)
            self.log_event(model_round, "start_aggregate")

            # Set the round timeout
            if self.settings.dfl.aggregation_timeout > 0:
                self.register_task("aggregate_%d_timeout" % model_round, self.on_aggregation_timeout, model_round,
                                   index, delay=self.settings.dfl.aggregation_timeout)

            self.aggregate_sample_estimate = index
            self.is_aggregating = True
            self.model_manager.reset_incoming_trained_models()
            self.aggregate_start_time = time.time()
            self.completed_aggregation = False
            self.model_manager.process_incoming_trained_model(peer_pk, model)
        elif index == self.aggregate_sample_estimate and not self.completed_aggregation:
            self.model_manager.process_incoming_trained_model(peer_pk, model)
        else:
            self.logger.info("Participant %s ignoring incoming trained model of round %d",
                             self.peer_manager.get_my_short_id(), model_round)
            return

        # Check whether we received enough incoming models
        if self.has_enough_trained_models():
            self.logger.info("Aggregator %s received sufficient trained models (%d) of round %d",
                             self.peer_manager.get_my_short_id(), len(self.model_manager.incoming_trained_models),
                             model_round)
            await self.aggregator_complete_round(model_round, index)
        else:
            self.logger.info("Aggregator %s has not enough trained models (%d) of round %d yet",
                             self.peer_manager.get_my_short_id(), len(self.model_manager.incoming_trained_models),
                             model_round)

    async def aggregator_complete_round(self, model_round: int, index: int):
        self.aggregate_start_time = 0
        self.completed_aggregation = True

        if not self.is_active:
            self.logger.warning("Aggregator %s completed aggregation but is offline!",
                                self.peer_manager.get_my_short_id())
            self.is_aggregating = False
            return

        timeout_task_name: str = "aggregate_%d_timeout" % model_round
        if self.is_pending_task_active(timeout_task_name):
            self.cancel_pending_task(timeout_task_name)

        if not self.model_manager.incoming_trained_models:
            self.logger.warning("Aggregator %s will complete round but has no models! Ignoring...",
                                self.peer_manager.get_my_short_id())
            self.is_aggregating = False
            return

        self.log_event(model_round, "done_aggregation")

        # 3.1. Aggregate these models
        self.logger.info("Aggregator %s will average the models of round %d",
                         self.peer_manager.get_my_short_id(), model_round)
        avg_model = self.model_manager.aggregate_trained_models()

        # 3.2. Remove these models from the model manager (they are not needed anymore)
        self.model_manager.reset_incoming_trained_models()

        # 3. Determine the participants of the next sample that are available
        participants = await self.determine_available_peers_for_sample(self.aggregate_sample_estimate,
                                                                       self.settings.dfl.sample_size)
        participants_ids: List[str] = [self.peer_manager.get_short_id(peer_id) for peer_id in participants]
        self.derived_samples.append((self.aggregate_sample_estimate, participants_ids))
        self.logger.info("Participant %s determined %d available participants for round %d: %s",
                         self.peer_manager.get_my_short_id(), len(participants_ids), model_round, participants_ids)

        # Is it still relevant what we're doing?
        if self.aggregate_sample_estimate > index:
            self.logger.warning("Work of participant %s for round %d not relevant anymore - stopping",
                                self.peer_manager.get_my_short_id(), model_round)
            self.is_aggregating = False
            return

        # 3.3. Distribute the average model to the available participants in the sample.
        task_name = "send_aggregated_model_%s" % self.aggregate_sample_estimate
        self.register_task(task_name, self.send_aggregated_model_to_participants, participants, avg_model,
                           self.aggregate_sample_estimate)

        # 4. Invoke the complete callback
        self.logger.info("Aggregator %s completed aggregation in round %d",
                         self.peer_manager.get_my_short_id(), model_round)
        if self.aggregate_complete_callback:
            ensure_future(self.aggregate_complete_callback(model_round, avg_model))

        self.is_aggregating = False

    def on_aggregation_timeout(self, model_round: int, index: int):
        self.logger.info("Aggregator %s triggered aggregation timeout in round %d - wrapping up",
                         self.peer_manager.get_my_short_id(), model_round)

        if index < self.aggregate_sample_estimate:
            self.logger.info("Aggregator %s triggered aggregation timeout in round %d but it's outdated - ignoring",
                             self.peer_manager.get_my_short_id(), model_round)
            return

        #if self.has_enough_trained_models_for_liveness():
        self.log_event(model_round, "aggregate_timeout")
        ensure_future(self.aggregator_complete_round(model_round, index))
        # else:
        #     self.aggregate_start_time = 0
        #     self.completed_aggregation = True
        #     self.is_aggregating = False
        #     self.logger.info("Aggregator %s triggered aggregation timeout in round %d but didn't receive sufficient "
        #                      "models to continue (%d models received)",
        #                      self.peer_manager.get_my_short_id(), model_round,
        #                      len(self.model_manager.incoming_trained_models))
        #     self.model_manager.reset_incoming_trained_models()

    def received_aggregated_model(self, peer: Peer, model_round: int, model: nn.Module) -> None:
        if self.shutting_down:
            self.logger.warning("Participant %s ignoring incoming aggregated model due to shutdown",
                                self.peer_manager.get_my_short_id())
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received aggregated model of round %d from aggregator %s",
                         self.peer_manager.get_my_short_id(), model_round - 1, peer_id)

        if model_round > self.train_sample_estimate:
            self.logger.info("Participant %s interrupting current training task for round %d",
                             self.peer_manager.get_my_short_id(), self.train_sample_estimate)
            if model_round > 1:  # We don't want to log this for the first round
                self.log_event(self.train_sample_estimate, "interrupt_training")
            self.train_sample_estimate = model_round
            self.cancel_current_training_task()
            self.completed_training = False
        if model_round == self.train_sample_estimate and not self.ongoing_training_task_name and not self.completed_training:
            self.model_manager.model = model
            self.train_in_round(model_round)
        else:
            self.logger.info("Participant %s NOT starting training round %d (train sample: %d, ongoing train task: %s, "
                             "completed training: %d)", self.peer_manager.get_my_short_id(), model_round,
                             self.train_sample_estimate, self.ongoing_training_task_name, self.completed_training)

    async def on_send_complete(self, result: TransferResult):
        await super().on_send_complete(result)
        self.peer_manager.update_peer_activity(result.peer.public_key.key_to_bin(), self.get_round_estimate())
