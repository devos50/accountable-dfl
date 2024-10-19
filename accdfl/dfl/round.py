from typing import Dict, Optional

from accdfl.dfl.reduction_manager import ReductionManager

from torch import nn


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Optional[nn.Module] = None

        # It could be that we receive chunks out of order, for example, before the round starts.
        # In that situation, store the chunk to process it later.
        self.out_of_order_chunks: Dict = {}

        self.reduction_manager: Optional[ReductionManager] = None

        # State
        self.is_training: bool = False
        self.train_done: bool = False
