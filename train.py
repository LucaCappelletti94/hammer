"""Executor for training a model."""

import silence_tensorflow.auto # pylint: disable=unused-import
from np_classifier.training import Trainer, SmilesDataset


def train():
    """Train the model."""
    dataset = SmilesDataset()
    trainer = Trainer(dataset)
    trainer.train()


if __name__ == "__main__":
    train()
