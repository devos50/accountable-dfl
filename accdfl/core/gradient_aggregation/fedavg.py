import copy
from typing import List

import torch
from torch import nn

from accdfl.core.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    @staticmethod
    def aggregate(models: List[nn.Module]):
        with torch.no_grad():
            weights = [float(1. / len(models)) for _ in range(len(models))]
            center_model = copy.deepcopy(models[0])
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model
