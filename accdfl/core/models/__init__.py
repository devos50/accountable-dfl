import pickle
from typing import Optional

import torch

from accdfl.core.models.Model import Model
from accdfl.core.models.resnet8 import ResNet8


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def unserialize_model(serialized_model: bytes, dataset: str, architecture: Optional[str] = None) -> torch.nn.Module:
    model = create_model(dataset, architecture=architecture)
    model.load_state_dict(pickle.loads(serialized_model))
    return model


def create_model(dataset: str, architecture: Optional[str] = None) -> Model:
    if dataset in ["shakespeare", "shakespeare_sub", "shakespeare_sub96"]:
        from accdfl.core.models.shakespeare import LSTM
        return LSTM()
    elif dataset == "cifar10" or dataset == "cifar10_niid":
        from accdfl.core.models.cifar10 import GNLeNet
        if not architecture:
            return GNLeNet(input_channel=3, output=10, model_input=(32, 32))
        elif architecture == "resnet8":
            return ResNet8()
        else:
            raise RuntimeError("Unknown model architecture for CIFAR10: %s" % architecture)
    elif dataset == "celeba":
        from accdfl.core.models.celeba import CNN
        return CNN()
    elif dataset == "femnist":
        from accdfl.core.models.femnist import CNN
        return CNN()
    elif dataset == "movielens":
        from accdfl.core.models.movielens import MatrixFactorization
        return MatrixFactorization()
    else:
        raise RuntimeError("Unknown dataset %s" % dataset)
