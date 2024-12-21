import logging
import os
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from accdfl.core.datasets import create_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import unserialize_model, serialize_model
from accdfl.core.session_settings import SessionSettings


DATASET = None


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model: Optional[nn.Module], settings: SessionSettings, participant_index: int):
        global DATASET
        self.model: nn.Module = model
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)

        # Keeps track of the incoming trained models as aggregator
        self.incoming_trained_models: Dict[bytes, nn.Module] = {}

        if not DATASET:
            DATASET = create_dataset(self.settings)

        self.model_trainer: ModelTrainer = ModelTrainer(DATASET, self.settings, self.participant_index)

    def process_incoming_trained_model(self, peer_pk: bytes, incoming_model: nn.Module):
        if peer_pk in self.incoming_trained_models:
            # We already processed this model
            return

        self.incoming_trained_models[peer_pk] = incoming_model

    def reset_incoming_trained_models(self):
        self.incoming_trained_models = {}

    @staticmethod
    def aggregate(models: List[nn.Module], weights: List[float]):
        if not weights:
            weights = [float(1. / len(models)) for _ in range(len(models))]
        else:
            assert len(weights) == len(models)

        with torch.no_grad():
            center_model = copy.deepcopy(models[0])
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model

    def aggregate_trained_models(self, weights: List[float] = None) -> Optional[nn.Module]:
        models = [model for model in self.incoming_trained_models.values()]
        return ModelManager.aggregate(models, weights=weights)

    async def train(self) -> int:
        samples_trained_on = await self.model_trainer.train(self.model)

        # Detach the gradients
        self.model = unserialize_model(serialize_model(self.model), self.settings.dataset, architecture=self.settings.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples_trained_on
