import os
from dataclasses import dataclass
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass
class LearningSettings:
    """
    Settings related to the learning process.
    """
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    local_steps: int


@dataclass
class ConfluxSettings:
    """
    Setting related to sample-based decentralized federated learning.
    """
    sample_size: int
    ping_timeout: float = 5
    chunks_in_sample: int = 10
    gossip_interval: float = 60


@dataclass_json
@dataclass
class SessionSettings:
    """
    All settings related to a training session.
    """
    work_dir: str
    dataset: str
    learning: LearningSettings
    participants: List[str]
    conflux_settings: Optional[ConfluxSettings] = None
    model: Optional[str] = None
    alpha: float = 1
    partitioner: str = "iid"  # iid, shards or dirichlet
    model_seed: int = 0
    train_device_name: str = "cpu"
