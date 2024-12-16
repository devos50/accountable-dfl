import asyncio
import json
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from math import floor
from random import Random
from typing import Dict, Optional, List, Tuple, Set

from accdfl.conflux.chunk_manager import ChunkManager
from accdfl.conflux.round import Round
import torch

import numpy as np

from ipv8.lazy_community import lazy_wrapper_wd
from ipv8.messaging.payload_headers import BinMemberAuthenticationPayload
from ipv8.types import Peer
from ipv8.util import succeed

from accdfl.core.community import LearningCommunity
from accdfl.core.models import create_model, serialize_chunk, serialize_model, unserialize_model
from accdfl.core.session_settings import SessionSettings
from accdfl.conflux.caches import PingPeersRequestCache, PingRequestCache
from accdfl.conflux.payloads import PingPayload, PongPayload, TrainDonePayload
from accdfl.conflux.sample_manager import SampleManager
from accdfl.util.transfer import TransferResult
from simulations.bandwidth_scheduler import BWScheduler


class ConfluxCommunity(LearningCommunity):
    community_id = unhexlify('d5889074c1e4c60423cee6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.random = Random(int.from_bytes(self.my_peer.public_key.key_to_bin(), 'big'))

        # Statistics
        self.active_peers_history = []
        self.bw_in_stats: Dict[str, Dict[str, int]] = {
            "bytes": {
                "model": 0,
                "ping": 0,
                "pong": 0,
            },
            "num": {
                "model": 0,
                "ping": 0,
                "pong": 0,
            }
        }

        self.bw_out_stats: Dict[str, Dict[str, int]] = {
            "bytes": {
                "model": 0,
                "ping": 0,
                "pong": 0,
            },
            "num": {
                "model": 0,
                "ping": 0,
                "pong": 0,
            }
        }
        self.determine_sample_durations = []
        self.events: List[Tuple[float, str, int, str]] = []

        self.round_info: Dict[int, Round] = {}
        self.last_round_completed: int = 0

        # State
        self.ongoing_training_task_name: Optional[str] = None

        # Components
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup

        self.other_nodes_bws: Dict[bytes, int] = {}

        self.nodes = None
        self.transfers: List[Tuple[str, str, int, float, float, str, bool]] = []
        self.bw_scheduler: BWScheduler = BWScheduler(self.my_peer.public_key.key_to_bin(), self.peer_manager.get_my_short_id())

        self.add_message_handler(PingPayload, self.on_ping)
        self.add_message_handler(PongPayload, self.on_pong)
        self.add_message_handler(TrainDonePayload, self.on_train_done_payload)

    def setup(self, settings: SessionSettings):
        self.logger.info("Setting up experiment with %d initial participants and sample size %d (I am participant %s)" %
                         (len(settings.participants), settings.conflux_settings.sample_size, self.peer_manager.get_my_short_id()))
        super().setup(settings)
        self.sample_manager = SampleManager(self.peer_manager, settings.conflux_settings.sample_size)

    def go_online(self):
        if self.is_active:
            self.logger.warning("Participant %s already online - ignoring", self.peer_manager.get_my_short_id())
            return

        super().go_online()

    def go_offline(self, graceful: bool = True) -> None:
        if not self.is_active:
            self.logger.warning("Participant %s already offline - ignoring", self.peer_manager.get_my_short_id())
            return

        super().go_offline()

        # Cancel training
        self.cancel_current_training_task()

        if not graceful:
            self.cancel_all_pending_tasks()

        self.bw_scheduler.kill_all_transfers()

    def determine_available_peers_for_sample(self, sample: int, count: int, pick_active_peers: bool = True) -> Future:
        if pick_active_peers:
            raw_peers = self.peer_manager.get_active_peers(sample)
        else:
            raw_peers = self.peer_manager.get_peers()
        candidate_peers = self.sample_manager.get_ordered_sample_list(sample, raw_peers)

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

        cache = PingRequestCache(self, ping_all_id, peer, self.settings.conflux_settings.ping_timeout)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def send_ping(self, peer: Peer, identifier: int) -> None:
        """
        Send a ping message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PingPayload(identifier)

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

        self.send_pong(peer, payload.identifier)

    def send_pong(self, peer: Peer, identifier: int) -> None:
        """
        Send a pong message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PongPayload(identifier)

        packet = self._ez_pack(self._prefix, PongPayload.msg_id, [auth, payload])
        self.bw_out_stats["bytes"]["pong"] += len(packet)
        self.bw_out_stats["num"]["pong"] += 1
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(PongPayload)
    def on_pong(self, peer: Peer, payload: PongPayload, raw_data: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        my_peer_id = self.peer_manager.get_my_short_id()
        peer_short_id = self.peer_manager.get_short_id(peer_pk)

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

        cache = self.request_cache.pop("ping-%s" % peer_short_id, payload.identifier)
        cache.on_pong()

    def send_train_done(self, participants: List[bytes], round_nr: Round) -> None:
        """
        Send a message to other nodes in the sample that training is done.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = TrainDonePayload(round_nr)

        for peer_pk in participants:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                continue

            packet = self._ez_pack(self._prefix, TrainDonePayload.msg_id, [auth, payload])
            self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(TrainDonePayload)
    def on_train_done_payload(self, peer: Peer, payload: PongPayload, raw_data: bytes) -> None:
        if payload.round not in self.round_info:
            self.round_info[payload.round] = Round(payload.round)

        round_info = self.round_info[payload.round]
        round_info.compute_done_acks_received += 1
        if round_info.compute_done_acks_received == self.settings.conflux_settings.sample_size:
            # We are now ready to start training in the next round!
            aggregated_model = round_info.chunk_manager_previous_sample.get_aggregated_model()
            round_info.model = aggregated_model
            self.train_in_round(round_info)

    def train_in_round(self, round_info: Round):
        self.ongoing_training_task_name = "round_%d" % round_info.round_nr
        if not self.is_pending_task_active(self.ongoing_training_task_name):
            task = self.register_task(self.ongoing_training_task_name, self.participate_in_round, round_info)

    async def participate_in_round(self, round_info: Round):
        """
        Participate in a round.
        """
        round_nr: int = round_info.round_nr
        if round_nr < 1:
            raise RuntimeError("Round number %d invalid!" % round_nr)

        self.logger.info("Participant %s starts training in round %d", self.peer_manager.get_my_short_id(), round_nr)

        if round_nr > 1 and self.round_complete_callback:
            ensure_future(self.round_complete_callback(round_nr - 1, round_info.model))

        # 1. Train the model
        round_info.is_training = True
        self.model_manager.model = round_info.model
        await self.model_manager.train()
        round_info.model = self.model_manager.model
        round_info.is_training = False
        round_info.train_done = True

        # It might be that we went offline at this point - check for it
        if not self.is_active:
            self.logger.warning("Participant %s went offline during model training in round %d - not proceeding",
                                self.peer_manager.get_my_short_id(), round_nr)
            return

        # 3. Start sharing the model chunks
        participants_next_sample = await self.determine_available_peers_for_sample(round_nr + 1, self.settings.conflux_settings.sample_size)
        participants_next_sample = sorted(participants_next_sample)
        await self.gossip_chunks(round_info, participants_next_sample)

        # 4. Let the nodes in the next sample know that we're done training
        self.send_train_done(participants_next_sample, round_info.round_nr + 1)

        # 5. Round complete!
        self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round_nr)
        self.last_round_completed = round_nr
        self.round_info.pop(round_nr)

    async def gossip_chunks(self, round_info: Round, participants: List[bytes]) -> None:
        self.logger.info("Participant %s starts gossiping chunks in round %d", self.peer_manager.get_my_short_id(), round_info.round_nr)
        round_info.chunk_manager_next_sample = ChunkManager(round, self.model_manager.model, self.settings.conflux_settings.chunks_in_sample)
        round_info.chunk_manager_next_sample.prepare()

        send_chunks_future = ensure_future(self.send_chunks(participants, round_info))

        await asyncio.sleep(self.settings.conflux_settings.gossip_interval)
        round_info.chunk_gossip_done = True
        round_info.chunk_manager_next_sample = None

        await send_chunks_future

    async def send_chunks(self, participants: List[bytes], round_info: Round) -> None:
        while True:
            # Send chunk
            # TODO we should probably start several transfers at the same time to better utilize outgoing bandwidth!
            idx, chunk = round_info.chunk_manager_next_sample.get_random_chunk_to_send(self.random)
            recipient_peer_pk = self.random.choice(participants)
            peer = self.get_peer_by_pk(recipient_peer_pk)
            if not peer:
                raise RuntimeError("Could not find peer with public key %s", hexlify(recipient_peer_pk).decode())

            await self.send_chunk(round_info.round_nr + 1, idx, chunk, peer)
            if round_info.chunk_gossip_done:
                break

    async def send_chunk(self, round: int, chunk_idx: int, chunk, peer, in_sample: bool = True):
        binary_data = serialize_chunk(chunk)
        response = {"round": round, "idx": chunk_idx, "type": "chunk", "in_sample": in_sample}
        return await self.send_data(binary_data, response, peer)

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

        if json_data["type"] == "chunk":
            if self.last_round_completed >= json_data["round"]:
                return

            incoming_chunk = torch.from_numpy(np.frombuffer(result.data, dtype=np.float32).copy())
            self.received_model_chunk(json_data["round"], json_data["idx"], incoming_chunk)
            return

        serialized_model = result.data[:json_data["model_data_len"]]
        self.bw_in_stats["bytes"]["model"] += len(serialized_model)
        self.bw_in_stats["num"]["model"] += 1
        incoming_model = unserialize_model(serialized_model, self.settings.dataset, architecture=self.settings.model)

        if json_data["type"] == "trained_model":
            await self.received_trained_model(result.peer, json_data["round"], incoming_model)
        elif json_data["type"] == "aggregated_model":
            self.received_aggregated_model(result.peer, json_data["round"], incoming_model)
        else:
            raise RuntimeError("Received unknown message type %s" % json_data["type"])

    def received_model_chunk(self, round_nr: int, chunk_idx: int, chunk) -> None:
        if round_nr not in self.round_info:
            # We received a chunk but haven't started this round yet - store it.
            new_round = Round(round_nr)
            self.round_info[round_nr] = new_round

            model = create_model(self.settings.dataset, architecture=self.settings.model)
            new_round.chunk_manager_previous_sample = ChunkManager(round_nr, model, self.settings.conflux_settings.chunks_in_sample)
            new_round.chunk_manager_previous_sample.process_received_chunk(chunk_idx, chunk)
        else:
            # Otherwise, process it right away!
            chunk_manager_previous_sample = self.round_info[round_nr].chunk_manager_previous_sample
            if not chunk_manager_previous_sample:
                model = create_model(self.settings.dataset, architecture=self.settings.model)
                self.round_info[round_nr].chunk_manager_previous_sample = ChunkManager(round_nr, model, self.settings.conflux_settings.chunks_in_sample)
            self.round_info[round_nr].chunk_manager_previous_sample.process_received_chunk(chunk_idx, chunk)

    async def send_data(self, binary_data: bytes, response: bytes, peer):
        serialized_response = json.dumps(response).encode()
        found: bool = False
        transfer_success: bool = True
        transfer_time: float = 0
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                transfer_start_time = asyncio.get_event_loop().time()
                if self.bw_scheduler.bw_limit > 0:
                    transfer_size: int = len(binary_data) + len(serialized_response)
                    transfer = self.bw_scheduler.add_transfer(node.overlays[0].bw_scheduler, transfer_size)
                    transfer.metadata = response
                    self.logger.info("Model transfer %s => %s started at t=%f",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     transfer_start_time)
                    try:
                        await transfer.complete_future
                    except RuntimeError:
                        transfer_success = False
                    transfer_time = asyncio.get_event_loop().time() - transfer_start_time

                    transferred_bytes: int = transfer.get_transferred_bytes()
                    self.endpoint.bytes_up += transferred_bytes
                    node.overlays[0].endpoint.bytes_down += transferred_bytes

                    self.logger.info("Model transfer %s => %s %s at t=%f and took %f s.",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     "completed" if transfer_success else "failed",
                                     transfer_start_time, transfer_time)
                else:
                    self.endpoint.bytes_up += len(binary_data) + len(serialized_response)
                    node.overlays[0].endpoint.bytes_down += len(binary_data) + len(serialized_response)

                json_data = json.loads(serialized_response.decode())
                self.transfers.append((self.peer_manager.get_my_short_id(),
                                       node.overlays[0].peer_manager.get_my_short_id(), json_data["round"],
                                       transfer_start_time, transfer_time, json_data["type"], transfer_success))

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)

        return transfer_success
