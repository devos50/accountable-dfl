from __future__ import annotations

import math
from typing import Iterable, Optional

from accdfl.util.eva.exceptions import SizeException, TransferException
from accdfl.util.eva.payload import Data
from accdfl.util.eva.result import TransferResult
from accdfl.util.eva.transfer.base import Transfer


class OutgoingTransfer(Transfer):
    def __init__(self, data: bytes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_size = len(data)

        limit = self.settings.binary_size_limit
        if self.data_size > limit:
            raise SizeException(f'Current data size limit {limit} has been exceeded: {self.data_size}')

        self.data = data
        self.block_count = math.ceil(self.data_size / self.settings.block_size)
        self.acknowledgement_received = False

    def on_acknowledgement(self, ack_number: int, window_size: int) -> Iterable[Data]:
        self.update()
        self.acknowledgement_received = True
        is_final_acknowledgement = ack_number > self.block_count
        if is_final_acknowledgement:
            result = TransferResult(peer=self.peer, info=self.info, data=self.data, nonce=self.nonce)
            self.finish(result=result)
            return

        for block_number in range(ack_number, ack_number + window_size):
            block = self._get_block(block_number)
            yield Data(block_number, self.nonce, block)
            if len(block) == 0:
                return

    def finish(self, *, result: Optional[TransferResult] = None, exception: Optional[TransferException] = None):
        self.container.pop(self.peer, None)
        super().finish(result=result, exception=exception)
        self.data = None

    def _get_block(self, number: int) -> bytes:
        start_position = number * self.settings.block_size
        stop_position = start_position + self.settings.block_size
        return self.data[start_position:stop_position]
