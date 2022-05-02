from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from accdfl.util.eva.transfer.base import Transfer


class TransferException(Exception):
    def __init__(self, message: str, transfer: Optional[Transfer] = None):
        super().__init__(message)
        self.transfer = transfer


class SizeException(TransferException):
    pass


class TimeoutException(TransferException):
    pass


class ValueException(TransferException):
    pass


class TransferLimitException(TransferException):
    """Maximum simultaneous transfers limit exceeded"""


class TransferCancelledException(TransferException):
    """Raised in the case that future was cancelled"""
