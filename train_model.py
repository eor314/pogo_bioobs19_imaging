#!/usr/bin/env python3
from torch.utils.data.sampler import Sampler
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch.optim import lr_scheduler
from torch.utils.data import RandomSampler
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, RandomCrop, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor)
from tqdm import tnrange, tqdm, tqdm_notebook

from utilities.display_utils import imshow_tensor
from utilities.split import stratified_random_split

from ignite.handlers import EarlyStopping

TRAINING_PATH = "/data1/mschroeder/Datasets/19-05-11 MiniZooScanNet/train"
VALIDATION_PATH = "/data1/mschroeder/Datasets/19-05-11 MiniZooScanNet/val"


class StratifiedSampler(Sampler):
    def __init__(self, targets, samples_per_class=1000):
        self.targets = np.array(targets)

        self.target_counts = np.clip(np.bincount(
            self.targets), None, samples_per_class)

        print("target counts", self.target_counts)

    def __iter__(self):
        indices = []
        for t, n in enumerate(self.target_counts):
            t_indices = np.flatnonzero(self.targets == t)

            if len(t_indices) < 1:
                continue

            replace = len(t_indices) < n
            t_indices = np.random.choice(t_indices, n, replace=replace)

            indices.append(t_indices)

        indices = np.concatenate(indices)

        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return int(self.target_counts.sum())


def run(max_epochs=100):
    transform = Compose([
        # Resize every image to a 224x244 square
        Resize((224, 224)),
        RandomCrop(224, 8, padding_mode="edge"),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        # Convert to a tensor that PyTorch can work with
        ToTensor()
    ])

    # Images are located at at {dataset_path}/{class_name}/{objid}.jpg
    dataset_train = ImageFolder(TRAINING_PATH, transform)
    dataset_val = ImageFolder(TRAINING_PATH, transform)

    # Make sure that the class names are identical
    assert dataset_train.classes == dataset_val.classes

    model = models.resnet18(pretrained=True)

    # get the number of features that are input to the fully connected layer
    num_ftrs = model.fc.in_features

    # reset the fully connect layer
    model.fc = nn.Linear(num_ftrs, len(dataset_train.classes))

    # Transfer model to GPU
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128,
                                               sampler=StratifiedSampler(dataset_train.targets), num_workers=4)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=128, num_workers=4, shuffle=True)

    trainer = create_supervised_trainer(
        model, optimizer, F.cross_entropy, device="cuda")
    train_evaluator = create_supervised_evaluator(model,
                                                  metrics={'accuracy': Accuracy(),
                                                           'nll': Loss(F.cross_entropy)},
                                                  device="cuda")

    val_evaluator = create_supervised_evaluator(model,
                                                metrics={'accuracy': Accuracy(),
                                                         'nll': Loss(F.cross_entropy)},
                                                device="cuda")

    desc = "ITERATION - loss: {:.2f}"
    log_interval = 1

    pbar = tqdm(
        initial=0, leave=False, total=len(loader_train),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(loader_train) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    # Display training metrics after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        print("Evaluating...")
        train_evaluator.run(loader_train)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    # Display validation metrics after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(loader_val)
        metrics = val_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss

    handler = EarlyStopping(
        patience=10, score_function=score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    handler = ModelCheckpoint('models', 'myprefix', create_dir=True,
                              n_saved=20, score_function=score_function, score_name="val_loss", save_as_state_dict=False)
    val_evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, handler, {'model': model})

    trainer.run(loader_train, max_epochs=max_epochs)

    pbar.close()


if __name__ == "__main__":
    run()
