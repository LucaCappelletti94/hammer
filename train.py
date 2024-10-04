"""Executor for training a model."""

import silence_tensorflow.auto  # pylint: disable=unused-import
import pandas as pd
import tensorflow as tf
from np_classifier.training import Trainer, Dataset


def train():
    """Train the model."""
    # First of all, we ensure that TensorFlow is using the GPU
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    dataset = Dataset()
    trainer = Trainer(dataset)

    performance: pd.DataFrame = trainer.holdouts()
    performance.to_csv("performance.csv", index=False)


if __name__ == "__main__":
    train()
