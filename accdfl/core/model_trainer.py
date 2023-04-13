import logging
import os
import time
from asyncio import sleep
from typing import Optional

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss

from accdfl.core.datasets import create_dataset, Dataset
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.session_settings import SessionSettings

AUGMENTATION_FACTOR_SIM = 3.0


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, data_dir, settings: SessionSettings, participant_index: int):
        """
        :param simulated_speed: compute speed of the simulated device, in ms/sample.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.simulated_speed: Optional[float] = None

        if settings.dataset in ["cifar10", "mnist", "movielens", "spambase"]:
            self.train_dir = data_dir
        else:
            self.train_dir = os.path.join(data_dir, "per_user_data", "train")
        self.dataset: Optional[Dataset] = None

    async def train(self, model, device_name: str = "cpu") -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        if not self.dataset:
            self.dataset = create_dataset(self.settings, participant_index=self.participant_index, train_dir=self.train_dir)

        device = torch.device(device_name)
        model.to(device)
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum, self.settings.learning.weight_decay)
        train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // self.settings.learning.batch_size
        if len(train_set.dataset) % self.settings.learning.batch_size != 0:
            local_steps += 1

        self.logger.info("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, wd: %f, "
                         "data points: %d)",
                         local_steps, device_name, self.settings.learning.batch_size,
                         self.settings.learning.learning_rate, self.settings.learning.weight_decay,
                         len(train_set.dataset))

        start_time = time.time()
        samples_trained_on = 0
        for local_step in range(local_steps):
            try:
                data, target = next(train_set_it)
                model.train()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                optimizer.optimizer.zero_grad()
                self.logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
                output = model.forward(data)
                samples_trained_on += len(data)

                if self.settings.dataset == "movielens":
                    lossf = MSELoss()
                elif self.settings.dataset == "cifar10":
                    if self.settings.model == "resnet8":
                        lossf = CrossEntropyLoss()
                    else:
                        lossf = NLLLoss()
                else:
                    lossf = CrossEntropyLoss()

                loss = lossf(output, target)
                self.logger.debug('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
                loss.backward()
                optimizer.optimizer.step()
            except StopIteration:
                pass

        if self.settings.is_simulation:
            # If we're running a simulation, we should advance the time of the DiscreteLoop with either the simulated
            # elapsed time or the elapsed real-world time for training. Otherwise,training would be considered instant
            # in our simulations.
            if self.simulated_speed:
                elapsed_time = AUGMENTATION_FACTOR_SIM * local_steps * self.settings.learning.batch_size * (self.simulated_speed / 1000)
            else:
                elapsed_time = time.time() - start_time

            self.logger.info("Model training took %f s.", elapsed_time)
            await sleep(elapsed_time)

        return samples_trained_on
