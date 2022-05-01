from __future__ import annotations

import asyncio
import logging
import time
from asyncio import Future
from typing import Dict, Optional, Type

from ipv8.types import Peer

from accdfl.util.eva.aliases import TransferCompleteCallback, TransferErrorCallback
from accdfl.util.eva.exceptions import TimeoutException, TransferException
from accdfl.util.eva.result import TransferResult
from accdfl.util.eva.settings import EVASettings


class Transfer:
    """The class describes an incoming or an outgoing transfer"""

    NONE = -1

    def __init__(self, container: Dict[Peer, Type[Transfer]], peer: Peer, info: bytes, nonce: int,
                 settings: EVASettings, data_size: int = 0, on_complete: Optional[TransferCompleteCallback] = None,
                 on_error: Optional[TransferErrorCallback] = None):
        """ This class has been used internally by the EVA protocol"""
        self.container = container
        self.peer = peer
        self.info = info
        self.data_size = data_size
        self.nonce = nonce
        self.on_complete = on_complete
        self.on_error = on_error
        self.settings = settings
        self.future = Future()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.updated = 0
        self.attempt = 0
        self.finished = False

    def update(self):
        self.updated = time.time()

    def finish(self, *, result: Optional[TransferResult] = None, exception: Optional[TransferException] = None):
        if self.finished:
            return

        if exception:
            self.logger.warning(f'Finish with exception: {exception.__class__.__name__}: {exception}|Peer: {self.peer}')
            self.future.set_exception(exception)

            # To prevent "Future exception was never retrieved" error when the future is not used
            self.future.exception()

            if self.on_error:
                asyncio.create_task(self.on_error(self.peer, exception))

        if result:
            self.logger.debug(f'Finish with result: {result}')
            self.future.set_result(result)
            if self.on_complete:
                asyncio.create_task(self.on_complete(result))

        self.finished = True
        self.container = None

    async def terminate_by_timeout_task(self):
        timeout = self.settings.timeout_interval_in_sec
        remaining_time = timeout

        while self.settings.terminate_by_timeout_enabled:
            await asyncio.sleep(remaining_time)
            if self.finished:
                return

            remaining_time = timeout - (time.time() - self.updated)
            if remaining_time <= 0:  # it is time to terminate
                exception = TimeoutException('Terminated by timeout', self)
                self.finish(exception=exception)
                return
