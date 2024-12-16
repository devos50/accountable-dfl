import logging
from binascii import hexlify
from typing import Dict, Optional, List, Tuple

from accdfl.core import NodeMembershipChange

NO_ACTIVITY_INFO = -1


class PeerManager:
    """
    The PeerManager keeps track of the population of peers that are participating in the training process.
    """

    def __init__(self, my_pk: bytes):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.my_pk: bytes = my_pk
        self.last_active: Dict[bytes, Tuple[int, Tuple[int, NodeMembershipChange]]] = {}
        self.last_active_pending: Dict[bytes, Tuple[int, Tuple[int, NodeMembershipChange]]] = {}  # Pending changes that are not immediately applied.

    def add_peer(self, peer_pk: bytes, round_active: Optional[int] = NO_ACTIVITY_INFO) -> None:
        """
        Add a new peer to this manager.
        :param peer_pk: The public key of the peer to add.
        :param round_active: The round that we last heard from this peer.
        """
        if peer_pk in self.last_active:
            return

        self.logger.debug("Participant %s adding participant %s to local view",
                          self.get_my_short_id(), self.get_short_id(peer_pk))
        self.last_active[peer_pk] = (round_active, (0, NodeMembershipChange.JOIN))

    def remove_peer(self, peer_pk) -> None:
        """
        Remove this peer from this manager.
        :param peer_pk: The public key of the peer to remove.
        """
        self.last_active.pop(peer_pk, None)

    def get_active_peers(self, round: Optional[int] = None) -> List[bytes]:
        active_peers = [peer_pk for peer_pk, status in self.last_active.items() if (status[1][1] != NodeMembershipChange.LEAVE or peer_pk == self.my_pk)]
        if round:
            active_peers = [peer_pk for peer_pk in active_peers]
        return active_peers

    def get_peers(self) -> List[bytes]:
        return [peer_pk for peer_pk, status in self.last_active.items()]

    def get_num_peers(self, round: Optional[int] = None) -> int:
        """
        Return the number of peers in the local view.
        """
        return len(self.get_active_peers(round))

    def update_peer_activity(self, peer_pk: bytes, round_active: int) -> None:
        """
        Update the status of a particular peer.
        :param peer_pk: The public key of the peer which activity status will be udpated.
        :param round_active: The last round in which this peer has been active.
        """
        info = self.last_active[peer_pk]
        self.last_active[peer_pk] = (max(self.last_active[peer_pk][0], round_active), info[1])

    def get_my_short_id(self) -> str:
        """
        Return a short description of your public key
        """
        return hexlify(self.my_pk).decode()[-8:]

    @staticmethod
    def get_short_id(peer_pk: bytes) -> str:
        return hexlify(peer_pk).decode()[-8:]

    def flush_last_active_pending(self) -> None:
        """
        Flush the pending changes to the population view.
        """
        if self.last_active_pending:
            self.logger.info("Participant %s flushing pending changes to population view", self.get_my_short_id())
            self.merge_population_views(self.last_active_pending)
            self.last_active_pending = {}

    def get_highest_round_in_population_view(self) -> int:
        """
        Return the highest round in the population view.
        """
        if not self.last_active:
            return -1
        return max([round for round, _ in self.last_active.values()])

    def merge_population_views(self, other_view: Dict[bytes, Tuple[int, Tuple[int, NodeMembershipChange]]]) -> None:
        """
        Reconcile the differences between two population views.
        """
        for peer_pk, info in other_view.items():
            # Is this a new joining node?
            if peer_pk not in self.last_active:
                # This seems to be a new node joining
                self.logger.info("Participant %s adds peer %s to local view",
                                 self.get_my_short_id(), self.get_short_id(peer_pk))
                self.last_active[peer_pk] = other_view[peer_pk]
                continue

            # This peer is already in the view - take its latest information

            # Check the last round activity
            last_round_active = info[0]
            if last_round_active > self.last_active[peer_pk][0]:
                self.update_peer_activity(peer_pk, last_round_active)

            # Update node membership status
            membership_index = info[1][0]
            if membership_index > self.last_active[peer_pk][1][0]:
                self.logger.debug("Participant %s updating membership status of participant %s to: %s",
                                  self.get_my_short_id(), self.get_short_id(peer_pk), str(info[1]))
                self.last_active[peer_pk] = (self.last_active[peer_pk][0], info[1])
