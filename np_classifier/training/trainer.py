"""Submodule handling the training of the multi-modal multi-class classifier."""

import tensorflow as tf
import pandas as pd
import compress_json
from np_classifier.training.smiles_dataset import Dataset
from np_classifier.training.model import Classifier


class Trainer:
    """Class handling the training of the multi-modal multi-class classifier."""

    def __init__(self, smiles_dataset: Dataset, number_of_epochs: int = 10_000):
        """Initialize the trainer."""
        assert isinstance(smiles_dataset, Dataset)
        self._smiles_dataset = smiles_dataset
        self._number_of_epochs = number_of_epochs
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def holdouts(
        self,
    ) -> pd.DataFrame:
        """Train the classifier."""
        _train, test = self._smiles_dataset.primary_split()
        all_performance = []
        for holdout_number, (sub_train, valid) in enumerate(
            self._smiles_dataset.train_split()
        ):
            classifier = Classifier(
                number_of_epochs=self._number_of_epochs,
            )
            classifier.train(sub_train, valid, holdout_number=holdout_number)
            sub_train_performance = classifier.evaluate(sub_train)
            sub_valid_performance = classifier.evaluate(valid)
            test_performance = classifier.evaluate(test)
            for performance, subset in [
                (sub_train_performance, "subtrain"),
                (sub_valid_performance, "validation"),
                (test_performance, "test"),
            ]:
                all_performance.append(
                    {
                        **performance,
                        "holdout": holdout_number,
                        "subset": subset,
                    }
                )
                compress_json.dump(
                    all_performance,
                    "current_performance.json",
                )
        return pd.DataFrame(all_performance)
