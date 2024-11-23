from asyncio import Future
from typing import List, Optional

from accdfl.dfl.chunk_manager import ChunkManager

from torch import nn


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Optional[nn.Module] = None

        self.chunk_manager_next_sample: Optional[ChunkManager] = None      # ChunkManager for sending chunks to the next sample
        self.chunk_manager_previous_sample: Optional[ChunkManager] = None  # ChunkManager for receiving chunks from the previous sample

        # State
        self.is_training: bool = False
        self.train_done: bool = False
        self.chunk_gossip_done: bool = False
        self.compute_done_acks_received: int = 0
        self.other_peers_ready_for_gossip: Future = Future()
