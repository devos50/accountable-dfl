import os

from accdfl.core.datasets import create_dataset

evaluator = None


def setup_evaluator(data_dir, parameters):
    global evaluator
    evaluator = ModelEvaluator(data_dir, parameters)


def evaluate_accuracy(model):
    global evaluator
    return evaluator.evaluate_accuracy(model)


class ModelEvaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    Runs in a separate process.
    """

    def __init__(self, data_dir, parameters):
        if parameters["dataset"] in ["cifar10", "mnist", "movielens"]:
            test_dir = data_dir
        else:
            test_dir = os.path.join(data_dir, "data", "test")
        self.dataset = create_dataset(parameters, test_dir=test_dir)

    def evaluate_accuracy(self, model):
        model.eval()
        return self.dataset.test(model)
