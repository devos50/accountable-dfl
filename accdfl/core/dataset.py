import logging
import math
from random import Random
from typing import Dict, List, Tuple

import torch

from torchvision import datasets, transforms


class Dataset:

    def __init__(self, data_dir, parameters: Dict, participant_index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_set = None
        self.validation_set = None
        self.data_dir = data_dir
        self.parameters = parameters
        self.batch_size = parameters["batch_size"]
        self.total_participants = len(parameters["participants"])
        self.participant_index = participant_index

        if parameters["dataset"] == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.dataset = datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=transform)
            self.test_dataset = datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=transform)
        elif parameters["dataset"] == "cifar10":
            transform = transforms.Compose([
                # transforms.Pad(4),
                # RandomHorizontalFlip(random=rand), # Reimplemented to be able to use a deterministic seed
                # RandomCrop(32, random=rand),       # Reimplemented to be able to use a deterministic seed
                transforms.ToTensor()])
            self.dataset = datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=transform)
            self.test_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=transform)

        self.iterator = None
        self.validation_iterator = None
        self.test_iterator = None
        self.reset_test_iterator()

        self.partition_dataset()

    def get_ranges(self) -> List[Tuple[int, int]]:
        rand = Random()
        rand.seed(1337)
        remaining_classes = [n for n in self.parameters["nodes_per_class"]]
        local_samples_per_class = [int(t / n) for t, n in zip(self.parameters["samples_per_class"], self.parameters["nodes_per_class"])]

        # Sample without putting the previous samples back
        # in the bag until empty, to guarantee coverage
        # of all classes with few nodes or rare classes
        def sampler_gen(remaining_classes, k):
            while sum(remaining_classes) > 0:
                choices = []
                max_class = max(remaining_classes)
                while len(choices) < k:
                    for c in range(10):
                        if remaining_classes[c] >= max_class:
                            choices.append(c)
                    max_class -= 1

                s = rand.sample(choices, k)
                for c in s:
                    remaining_classes[c] -= 1
                yield s

        def classes():
            samples = next(sampler)
            classes = [0. for c in range(10)]
            for i in range(self.parameters["local_classes"]):
                classes[samples[i]] = 1.
            return classes

        sampler = sampler_gen(remaining_classes, self.parameters["local_classes"])
        nodes = [{"classes": classes()} for _ in range(len(self.parameters["participants"]))]
        multiples = [0 for _ in range(10)]
        for n in nodes:
            for c in range(10):
                multiples[c] += n["classes"][c]

        logging.info('assign_classes: classes represented times {}'.format(multiples))

        # save [start, end[ for each class of every node where:
        # 'start' is the inclusive start index
        # 'end' is the exclusive end index
        start = [0 for i in range(10)]
        for n in nodes:
            end = [start[c] + int(n["classes"][c] * local_samples_per_class[c])
                   for c in range(10)]
            n['samples'] = [(start[c], end[c]) for c in range(10)]
            start = end

        return nodes[self.participant_index]["samples"]

    def partition_dataset(self) -> None:
        # Generate the ranges of samples for this particular node
        ranges = self.get_ranges()

        # Partition the dataset, based on the participant index
        # TODO assume iid distribution + hard-coded values
        samples_per_class_per_node = [t // self.total_participants for t in self.parameters["samples_per_class"]]
        rand = Random()
        rand.seed(1337)

        logging.info('partition: split the dataset per class (samples per class: %s)', samples_per_class_per_node)
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
        partition = []
        for c in range(10):
            start, end = tuple(ranges[c])
            partition.extend(shuffled[c][start:end])

        self.train_set = [self.dataset[i] for i in partition]
        # TODO the validation set is currently the same as the training set!
        self.validation_set = [self.dataset[i] for i in partition]
        self.reset_train_iterator()
        self.reset_validation_iterator()

        self.logger.info("Partition: done")

    def reset_train_iterator(self):
        self.iterator = iter(torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        ))

    def reset_validation_iterator(self):
        self.validation_iterator = iter(torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            shuffle=True
        ))

    def reset_test_iterator(self):
        self.test_iterator = iter(torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True
        ))

    def get_statistics(self) -> Dict:
        samples_per_class = [0] * 10
        for data, target in self.train_set:
            samples_per_class[target] += 1
        return {
            "total_samples": len(self.train_set),
            "samples_per_class": samples_per_class
        }
