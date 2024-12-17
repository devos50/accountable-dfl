import asyncio
import time
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from typing import Optional, Callable, Dict, List

import torch

from accdfl.core.models import create_model
from accdfl.core.model_manager import ModelManager
from accdfl.core.peer_manager import PeerManager
from accdfl.core.session_settings import SessionSettings
from accdfl.util.transfer import TransferResult

from ipv8.community import Community
from ipv8.requestcache import RequestCache
from ipv8.types import Peer


class LearningCommunity(Community):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_cache = RequestCache()
        self.my_id = self.my_peer.public_key.key_to_bin()
        self.round_complete_callback: Optional[Callable] = None

        self.peers_list: List[Peer] = []

        # Settings
        self.settings: Optional[SessionSettings] = None

        # State
        self.is_active = False
        self.did_setup = False
        self.shutting_down = False

        # Components
        self.peer_manager: PeerManager = PeerManager(self.my_id)
        self.model_manager: Optional[ModelManager] = None    # Initialized when the process is setup

        # Availability traces
        self.traces: Optional[Dict] = None
        self.traces_count: int = 0

        self.logger.info("The %s started with peer ID: %s", self.__class__.__name__,
                         self.peer_manager.get_my_short_id())

    def start(self):
        """
        Start to participate in the training process.
        """
        assert self.did_setup, "Process has not been setup - call setup() first"
        self.is_active = True

    def set_traces(self, traces: Dict) -> None:
        self.traces = traces
        events: int = 0

        # Schedule the join/leave events
        for active_timestamp in self.traces["active"]:
            if active_timestamp == 0:
                continue  # We assume peers will be online at t=0

            self.register_anonymous_task("join", self.go_online, delay=active_timestamp)
            events += 1

        for inactive_timestamp in self.traces["inactive"]:
            self.register_anonymous_task("leave", self.go_offline, delay=inactive_timestamp)
            events += 1

        self.logger.info("Scheduled %d join/leave events for peer %s (trace length in sec: %d)", events,
                         self.peer_manager.get_my_short_id(), traces["finish_time"])

        # Schedule the next call to set_traces
        self.register_task("reapply-trace-%s-%d" % (self.peer_manager.get_my_short_id(), self.traces_count),
                           self.set_traces, self.traces, delay=self.traces["finish_time"])
        self.traces_count += 1

    def go_online(self):
        self.is_active = True
        cur_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        self.logger.info("Participant %s comes online (t=%d)", self.peer_manager.get_my_short_id(), cur_time)

    def go_offline(self, graceful: bool = True):
        self.is_active = False
        cur_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        self.logger.info("Participant %s will go offline (t=%d)", self.peer_manager.get_my_short_id(), cur_time)

    def setup(self, settings: SessionSettings):
        self.settings = settings
        for participant in settings.participants:
            self.peer_manager.add_peer(unhexlify(participant))

        # Initialize the model
        torch.manual_seed(settings.model_seed)
        model = create_model(settings.dataset, architecture=settings.model)
        participant_index = settings.participants.index(hexlify(self.my_id).decode())
        self.model_manager = ModelManager(model, settings, participant_index)

        self.did_setup = True

    def get_peers(self):
        if self.peers_list:
            return self.peers_list
        return super().get_peers()

    def get_peer_by_pk(self, target_pk: bytes):
        peers = list(self.get_peers())
        for peer in peers:
            if peer.public_key.key_to_bin() == target_pk:
                return peer
        return None

    async def on_send_complete(self, result: TransferResult):
        peer_id = self.peer_manager.get_short_id(result.peer.public_key.key_to_bin())
        my_peer_id = self.peer_manager.get_my_short_id()
        self.logger.info(f'Outgoing transfer {my_peer_id} -> {peer_id} has completed: {result.info.decode()}')

    async def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')

    async def unload(self):
        self.shutting_down = True
        await self.request_cache.shutdown()
        await super().unload()
