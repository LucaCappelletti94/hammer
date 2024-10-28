"""Submodule handling the training of the multi-modal multi-class classifier."""

import os
import gc
from typing import Optional
import pandas as pd
from keras.api.backend import clear_session # type: ignore
from dict_hash import sha256, Hashable
from hammer.datasets import Dataset
from hammer.model import Hammer
from hammer.feature_settings import FeatureSettings
from hammer.augmentation_settings import AugmentationSettings


class Trainer(Hashable):
    """Class handling the training of the multi-modal multi-class classifier."""

    def __init__(
        self,
        dataset: Dataset,
        feature_settings: FeatureSettings,
        augmentation_settings: AugmentationSettings,
        maximal_number_of_epochs: int = 10_000,
        training_directory: str = "training",
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ):
        """Initialize the trainer."""
        assert isinstance(dataset, Dataset)
        self._dataset = dataset
        self._maximal_number_of_epochs = maximal_number_of_epochs
        self._feature_settings = feature_settings
        self._augmentation_settings = augmentation_settings
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._training_directory = training_directory

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return consistent hash of the current object."""
        return sha256(
            {
                "dataset": self._dataset,
                "maximal_number_of_epochs": self._maximal_number_of_epochs,
                "feature_settings": self._feature_settings,
                "augmentation_settings": self._augmentation_settings,
                "training_directory": self._training_directory,
            },
            use_approximation=use_approximation,
        )

    def train(
        self,
        test_size: float,
    ) -> dict[str, dict[str, float]]:
        """Train the classifier."""
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Restrict TensorFlow to only use the first GPU
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Visible devices must be set before GPUs have been initialized
                    print(e)
        except ImportError:
            pass

        (train_smiles, train_labels), (test_smiles, test_labels) = (
            self._dataset.primary_split(
                test_size=test_size,
            )
        )

        classifier = Hammer(
            dag=self._dataset.layered_dag(),
            feature_settings=self._feature_settings,
            scalers={},
            verbose=self._verbose,
            n_jobs=self._n_jobs,
        )
        classifier.fit(
            train_smiles=train_smiles,
            train_labels=train_labels,
            validation_smiles=test_smiles,
            validation_labels=test_labels,
            augmentation_settings=self._augmentation_settings,
            maximal_number_of_epochs=self._maximal_number_of_epochs,
        )
        classifier.save(self._training_directory)

        performance = {
            "train": classifier.evaluate(train_smiles, train_labels),
            "valid": classifier.evaluate(test_smiles, test_labels),
        }

        del classifier
        clear_session()
        gc.collect()

        return performance

    def holdouts(
        self,
        number_of_holdouts: int,
        test_size: float,
        validation_size: float,
    ) -> pd.DataFrame:
        """Train the classifier."""
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Restrict TensorFlow to only use the first GPU
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Visible devices must be set before GPUs have been initialized
                    print(e)
        except ImportError:
            pass

        (_train_smiles, _train_labels), (test_smiles, test_labels) = (
            self._dataset.primary_split(
                test_size=test_size,
            )
        )
        all_performance = []
        for holdout_number, (
            (sub_train_smiles, sub_train_labels),
            (validation_smiles, validation_labels),
        ) in enumerate(
            self._dataset.train_split(
                number_of_holdouts=number_of_holdouts,
                validation_size=validation_size,
                test_size=test_size,
            )
        ):
            holdout_hash = sha256(
                {
                    "self": self,
                    "sub_train_smiles": sub_train_smiles,
                    "sub_train_labels": sub_train_labels,
                    "validation_smiles": validation_smiles,
                    "validation_labels": validation_labels,
                    "test_smiles": test_smiles,
                    "test_labels": test_labels,
                },
                use_approximation=True,
            )

            path = os.path.join(self._training_directory, holdout_hash)

            if not os.path.exists(path):
                classifier = Hammer(
                    dag=self._dataset.layered_dag(),
                    feature_settings=self._feature_settings,
                    scalers={},
                    verbose=self._verbose,
                    n_jobs=self._n_jobs,
                )
                classifier.fit(
                    train_smiles=sub_train_smiles,
                    train_labels=sub_train_labels,
                    validation_smiles=validation_smiles,
                    validation_labels=validation_labels,
                    augmentation_settings=self._augmentation_settings,
                    maximal_number_of_epochs=self._maximal_number_of_epochs,
                )
                classifier.save(path)

            classifier = Hammer.load_from_path(path)

            sub_train_performance, sub_train_label_wise_performance = classifier.evaluate(
                sub_train_smiles, sub_train_labels
            )
            valid_performance, valid_label_wise_performance = classifier.evaluate(
                validation_smiles, validation_labels
            )
            test_performance, test_label_wise_performance  = classifier.evaluate(test_smiles, test_labels)
            for performance, label_wise_performance, subset in [
                (sub_train_performance, sub_train_label_wise_performance, "subtrain"),
                (valid_performance, valid_label_wise_performance, "validation"),
                (test_performance, test_label_wise_performance, "test"),
            ]:
                all_performance.append(
                    {
                        **performance,
                        "holdout": holdout_number,
                        "subset": subset,
                        "label": "Comulative"
                    }
                )
                for label, label_performance in label_wise_performance.items():
                    all_performance.append(
                        {
                            **label_performance,
                            "holdout": holdout_number,
                            "subset": subset,
                            "label": label,
                        }
                    )

            del classifier
            clear_session()
            gc.collect()

        return pd.DataFrame(all_performance)
