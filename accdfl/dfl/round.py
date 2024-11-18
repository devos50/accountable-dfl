from asyncio import Future
from typing import List, Optional

from accdfl.dfl.chunk_manager import ChunkManager

from torch import nn


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Optional[nn.Module] = None

        # It could be that we receive chunks out of order, for example, before the round starts.
        # In that situation, store the chunks to process it later.
        self.out_of_order_in_sample_chunks: List = []

        self.chunk_manager_in_sample: Optional[ChunkManager] = None        # ChunkManager for the in-sample aggregation
        self.chunk_manager_next_sample: Optional[ChunkManager] = None      # ChunkManager for sending chunks to the next sample
        self.chunk_manager_previous_sample: Optional[ChunkManager] = None  # ChunkManager for receiving chunks from the previous sample

        # State
        self.is_training: bool = False
        self.train_done: bool = False
        self.chunk_gossip_done: bool = False
        self.compute_done_acks_received: int = 0
        self.other_peers_ready_for_gossip: Future = Future()
