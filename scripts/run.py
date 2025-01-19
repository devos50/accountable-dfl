import logging
import os
import time

import torch

from accdfl.core.datasets import create_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings


logger = logging.getLogger("standalone-trainer")
ROUNDS = 1000


async def run(args):
    args.peers = 1

    learning_settings = LearningSettings(
        local_steps=args.local_steps,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    settings = SessionSettings(
        dataset=args.dataset,
        partitioner=args.partitioner,
        alpha=args.alpha,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        model=args.model,
    )

    data_path = os.path.join("data", "%s_n_%d" % (args.dataset, args.peers))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,learning_rate,accuracy,loss\n")

    await train_local(args, settings, data_path)



async def train_local(args, settings: SessionSettings, data_path: str):
    dataset = create_dataset(settings)
    evaluator = ModelEvaluator(dataset, settings)

    model = create_model(settings.dataset, architecture=settings.model)
    trainer = ModelTrainer(dataset, settings, 0)
    for round in range(1, ROUNDS + 1):
        start_time = time.time()
        print("Starting training round %d for peer %d" % (round, 0))
        await trainer.train(model)
        print("Training round %d for peer %d done - time: %f" % (round, 0, time.time() - start_time))

        if round % 10 == 0:
            acc, loss = evaluator.evaluate_accuracy(model)
            print("Accuracy: %f, loss: %f" % (acc, loss))

            with open(os.path.join(data_path, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,%s,%d,%d,%d,%f,%f,%f\n" % (args.dataset, "standalone", 0, args.peers, round, settings.learning.learning_rate, acc, loss))
