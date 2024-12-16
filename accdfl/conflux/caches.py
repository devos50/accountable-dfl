import logging
import time
from asyncio import get_event_loop, Future
from typing import List

from ipv8.requestcache import RandomNumberCache, NumberCache
from ipv8.types import Peer

PING_INTERVAL = 1.0


class PingRequestCache(NumberCache):
    """
    This cache is used to determine the availability status of a particular peer.
    """

    def __init__(self, community, ping_all_id: int, peer: Peer, ping_timeout: float):
        peer_short_id = community.peer_manager.get_short_id(peer.public_key.key_to_bin())
        super().__init__(community.request_cache, "ping-%s" % peer_short_id, ping_all_id)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.community = community
        self.peer = peer
        self.ping_all_id = ping_all_id
        self.request_cache = community.request_cache
        self.ping_timeout = ping_timeout
        self.num_pings = 0
        self.future = Future()
        self.task_name = None

    def start(self):
        self.send_ping()
        peer_id = self.community.peer_manager.get_short_id(self.peer.public_key.key_to_bin())
        self.task_name = "ping_%s_%s" % (self.number, peer_id)
        self.community.register_task(self.task_name, self.on_interval, delay=PING_INTERVAL)

    def send_ping(self):
        self.num_pings += 1
        peer_id = self.community.peer_manager.get_short_id(self.peer.public_key.key_to_bin())
        self.logger.debug("Sending ping %d to participant %s", self.num_pings, peer_id)
        self.community.send_ping(self.peer, self.number)

    def on_interval(self):
        if self.future.done():
            return

        self.send_ping()

    def on_pong(self):
        if self.task_name:
            self.community.cancel_pending_task(self.task_name)
        self.future.set_result((self.peer.public_key.key_to_bin(), True))

    @property
    def timeout_delay(self) -> float:
        return self.ping_timeout

    def on_timeout(self):
        if self.task_name:
            self.community.cancel_pending_task(self.task_name)
        self.future.set_result((self.peer.public_key.key_to_bin(), False))


class PingPeersRequestCache(RandomNumberCache):

    def __init__(self, community, peers: List[bytes], target_available_peers: int, round: int):
        super().__init__(community.request_cache, "ping-peers")
        self.community = community
        self.peers = peers
        self.target_available_peers = target_available_peers
        self.round = round
        self.available_peers = []
        self.next_peer_index = 0
        self.ping_timeout = self.community.settings.conflux_settings.ping_timeout
        self.future = Future()
        self.start_time = get_event_loop().time()

    @property
    def timeout_delay(self) -> float:
        # This cache will practically not timeout.
        return 3600.0

    def start(self):
        # Ping the first peers
        for peer_pk in self.peers[:self.target_available_peers]:
            self.ping_peer(peer_pk)
            self.next_peer_index += 1

    def finish(self):
        if self.community.request_cache.has(self.prefix, self._number):
            self.community.request_cache.pop(self.prefix, self._number)
        sample_duration = get_event_loop().time()
        self.community.determine_sample_durations.append((self.start_time, sample_duration))
        self.future.set_result(self.available_peers)

    def add_available_peer(self, peer_pk):
        self.available_peers.append(peer_pk)
        if len(self.available_peers) == self.target_available_peers and not self.future.done():
            self.finish()

    def on_pong_response(self, future):
        peer_pk, online = future.result()
        if online:
            self.add_available_peer(peer_pk)
        else:
            # Seems this peer was not online - try the next peer if available
            if self.next_peer_index < len(self.peers):
                self.ping_peer(self.peers[self.next_peer_index])
                self.next_peer_index += 1
            elif not self.future.done():
                # We're out of peers - return the available peers, it's the best we can do
                self.finish()

    def ping_peer(self, peer_pk: bytes):
        if peer_pk == self.community.my_id:
            self.add_available_peer(peer_pk)
            return

        # Otherwise, queue a ping operation to this peer
        future: Future = self.community.ping_peer(self._number, peer_pk)
        future.add_done_callback(self.on_pong_response)
