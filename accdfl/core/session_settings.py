import os
from dataclasses import dataclass
from typing import List, Optional, Type

from dataclasses_json import dataclass_json

from accdfl.core import TransmissionMethod
from accdfl.core.gradient_aggregation import GradientAggregationMethod


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
class DFLSettings:
    """
    Setting related to sample-based decentralized federated learning.
    """
    sample_size: int
    num_aggregators: int
    ping_timeout: float = 5
    inactivity_threshold: int = 50
    fixed_aggregator: Optional[bytes] = None
    aggregation_timeout: float = 300
    chunks_in_sample: int = 10
    gossip_interval: float = 60


@dataclass
class DLSettings:
    """
    Setting related to decentralized learning.
    """
    topology: str = "exp-one-peer"


@dataclass
class GLSettings:
    """
    Setting related to gossip learning.
    """
    round_timeout: float = 60


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
    all_participants: List[str]
    target_participants: int
    dataset_base_path: str = None
    dfl: Optional[DFLSettings] = None
    dl: Optional[DLSettings] = None
    gl: Optional[GLSettings] = None
    model: Optional[str] = None
    alpha: float = 1
    partitioner: str = "iid"  # iid, shards or dirichlet
    gradient_aggregation: GradientAggregationMethod = GradientAggregationMethod.FEDAVG
    model_seed: int = 0
    model_send_delay: float = 1.0
    transmission_method: TransmissionMethod = TransmissionMethod.EVA
    eva_block_size: int = 60000  # This value is extremely high and tuned for the DAS6
    eva_max_simultaneous_transfers: int = 30  # Corresponds to a peak usage of ~3.4 MB/s for an aggregator
    is_simulation: bool = False
    train_device_name: str = "cpu"
    bypass_training: bool = False  # Whether to bypass model training, can be useful to observe network dynamics
    rounds: int = 0


def dump_settings(settings: SessionSettings):
    """
    Dump the session settings if they do not exist yet.
    """
    settings_file_path = os.path.join(settings.work_dir, "settings.json")
    if not os.path.exists(settings_file_path):
        with open(settings_file_path, "w") as settings_file:
            settings_file.write(settings.to_json())
