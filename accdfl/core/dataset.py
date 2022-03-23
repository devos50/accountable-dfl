import logging
import math
from random import Random

import torch

from torchvision import datasets, transforms


class Dataset:

    def __init__(self, data_dir, batch_size, total_samples_per_class: int,
                 total_participants: int, participant_index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_set = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.total_participants = total_participants
        self.participant_index = participant_index
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform)
        self.iterator = None

        self.partition_dataset([total_samples_per_class] * 10)

    def partition_dataset(self, total_samples_per_class):
        # Partition the dataset, based on the participant index
        # TODO assume iid distribution + hard-coded values
        samples_per_class_per_node = [t / self.total_participants for t in total_samples_per_class]
        rand = Random()
        rand.seed(1337)

        logging.info('partition: split the dataset per class (samples per class %d', total_samples_per_class)
        indexes = {x: [] for x in range(10)}
        if type(self.dataset.targets) != torch.Tensor:
            targets = torch.tensor(self.dataset.targets)
        else:
            targets = self.dataset.targets
        for x in indexes:
            c = (targets.clone().detach() == x).nonzero()
            indexes[x] = c.view(len(c)).tolist()

        # We shuffle the list of indexes for each class so that a range of indexes
        # from the shuffled list corresponds to a random sample (without
        # replacement) from the list of examples.  This makes sampling faster in
        # the next step.
        #
        # Additionally, we append additional and different shufflings of the same
        # list of examples to cover the total number of examples assigned
        # when it is larger than the number of available examples.
        self.logger.info('partition: shuffling examples')
        shuffled = []
        for c in range(10):
            ind_len = len(indexes[c])
            min_len = ind_len
            shuffled_c = []
            for i in range(int(math.ceil(min_len / ind_len))):
                shuffled_c.extend(rand.sample(indexes[c], ind_len))
            shuffled.append(shuffled_c)

        # Sampling examples for each node now simply corresponds to extracting
        # the assigned range of examples for that node.
        self.logger.info('partition: sampling examples for each node')
        ranges = [[int(samples_per_class_per_node[cls_ind] * self.participant_index),
                   int(samples_per_class_per_node[cls_ind] * (self.participant_index + 1))] for cls_ind in range(10)]
        partition = []
        for c in range(10):
            start, end = tuple(ranges[c])
            partition.extend(shuffled[c][start:end])

        self.train_set = [self.dataset[i] for i in partition]
        self.reset_iterator()

        self.logger.info("Partition: done")

    def reset_iterator(self):
        self.iterator = iter(torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        ))
