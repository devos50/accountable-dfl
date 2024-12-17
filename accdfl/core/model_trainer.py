import logging
from asyncio import sleep, CancelledError, get_event_loop
from typing import Optional

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.utils.data import DataLoader

from accdfl.core.session_settings import SessionSettings

from flwr_datasets import FederatedDataset

from datasets import Dataset

AUGMENTATION_FACTOR_SIM = 3.0


class ModelTrainer:
    """
    Manager to train a particular model.
    """

    def __init__(self, dataset: FederatedDataset, settings: SessionSettings, participant_index: int):
        """
        :param simulated_speed: compute speed of the simulated device, in ms/sample.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.simulated_speed: Optional[float] = None
        self.total_training_time: float = 0
        self.is_training: bool = False
        self.dataset: FederatedDataset = dataset
        self.partition: Optional[Dataset] = None

    async def train(self, model, device_name: str = "cpu") -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        self.is_training = True
        local_steps = self.settings.learning.local_steps

        # Load the partition if it's not loaded yet
        if not self.partition:
            self.partition = self.dataset.load_partition(self.participant_index, "train")
            if(self.settings.dataset == "cifar10"):
                from accdfl.core.datasets.transforms import apply_transforms_cifar10 as transforms
                self.partition = self.partition.with_transform(transforms)
            else:
                raise RuntimeError("Unknown dataset %s for partitioning!" % self.settings.dataset)

        train_loader = DataLoader(self.partition, batch_size=self.settings.learning.batch_size, shuffle=True)

        device = torch.device(device_name)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.settings.learning.learning_rate,
            momentum=self.settings.learning.momentum,
            weight_decay=self.settings.learning.weight_decay)

        self.logger.info("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, wd: %f)",
                         local_steps, device_name, self.settings.learning.batch_size,
                         self.settings.learning.learning_rate, self.settings.learning.weight_decay)

        # If we're running a simulation, we should advance the time of the DiscreteLoop with either the simulated
        # elapsed time or the elapsed real-world time for training. Otherwise,training would be considered instant
        # in our simulations. We do this before the actual training so if our sleep gets interrupted, the local
        # model will not be updated.
        start_time = get_event_loop().time()
        if self.simulated_speed:
            elapsed_time = AUGMENTATION_FACTOR_SIM * local_steps * (self.simulated_speed / 1000)
        else:
            elapsed_time = 0

        try:
            await sleep(elapsed_time)
        except CancelledError:
            self.is_training = False
            self.total_training_time += (get_event_loop().time() - start_time)
            return 0  # Training got interrupted - don't update the model
        self.total_training_time += elapsed_time

        self.logger.info("Model training completed and took %f s.", elapsed_time)

        samples_trained_on = 0
        model = model.to(device)
        for local_step, batch in enumerate(train_loader):
            data, target = batch["img"], batch["label"]  # TODO hard-coded, not generic enough for different datasets
            if local_step >= local_steps:
                break

            model.train()
            data, target = Variable(data.to(device)), Variable(target.to(device))
            samples_trained_on += len(data)

            optimizer.zero_grad()
            self.logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
            output = model.forward(data)

            if self.settings.dataset == "movielens":
                lossf = MSELoss()
            elif self.settings.dataset == "cifar10":
                if self.settings.model in ["resnet8", "resnet18", "mobilenet_v3_large"]:
                    lossf = CrossEntropyLoss()
                else:
                    lossf = NLLLoss()
            else:
                lossf = CrossEntropyLoss()

            loss = lossf(output, target)
            self.logger.debug('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
            loss.backward()
            optimizer.step()

        self.is_training = False
        model.to("cpu")

        return samples_trained_on
