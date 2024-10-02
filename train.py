"""Executor for training a model."""

import silence_tensorflow.auto  # pylint: disable=unused-import
import pandas as pd
import compress_json
from np_classifier.training import Trainer, Dataset


def train():
    """Train the model."""
    dataset = Dataset()

    # We store to jsons the current pathway, superclasses, and classes
    compress_json.dump(dataset.pathway_names, "pathway_names.json")
    compress_json.dump(dataset.superclass_names, "superclass_names.json")
    compress_json.dump(dataset.class_names, "class_names.json")

    trainer = Trainer(dataset)

    performance: pd.DataFrame = trainer.holdouts()
    performance.to_csv("performance.csv", index=False)


if __name__ == "__main__":
    train()
