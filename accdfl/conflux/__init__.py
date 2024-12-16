"""
Contains code related to the DFL algorithm.
"""
from enum import Enum


class State(Enum):
    IDLE = 0
    TRAINING = 1
    AGGREGATING = 2
