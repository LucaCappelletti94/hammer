"""Submodule handling the training of the multi-modal multi-class classifier."""

import os
import gc
from typing import Optional
import pandas as pd
from keras.api.backend import clear_session  # type: ignore
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
        clear_session()
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

        (train_samples, train_labels), (test_samples, test_labels) = (
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
            train_samples=train_samples,
            train_labels=train_labels,
            validation_samples=test_samples,
            validation_labels=test_labels,
            augmentation_settings=self._augmentation_settings,
            maximal_number_of_epochs=self._maximal_number_of_epochs,
        )
        classifier.save(self._training_directory)

        performance = {
            "train": classifier.evaluate(train_samples, train_labels),
            "valid": classifier.evaluate(test_samples, test_labels),
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

        (_train_samples, _train_labels), (test_samples, test_labels) = (
            self._dataset.primary_split(
                test_size=test_size,
            )
        )
        all_performance = []
        for holdout_number, (
            (sub_train_samples, sub_train_labels),
            (validation_samples, validation_labels),
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
                    "sub_train_samples": sub_train_samples[:50],
                    "sub_train_labels": sub_train_labels[:50],
                    "validation_samples": validation_samples[:50],
                    "validation_labels": validation_labels[:50],
                    "test_samples": test_samples[:50],
                    "test_labels": test_labels[:50],
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
                history = classifier.fit(
                    train_samples=sub_train_samples,
                    train_labels=sub_train_labels,
                    validation_samples=validation_samples,
                    validation_labels=validation_labels,
                    augmentation_settings=self._augmentation_settings,
                    maximal_number_of_epochs=self._maximal_number_of_epochs,
                )

                history_df = pd.DataFrame(history.history)

                os.makedirs(
                    os.path.join(self._training_directory, holdout_hash), exist_ok=True
                )

                history_df.to_csv(
                    os.path.join(self._training_directory, holdout_hash, "history.csv"),
                    index=False,
                )

                classifier.save(path)

            classifier = Hammer.load_from_path(path)

            sub_train_performance = classifier.evaluate(
                sub_train_samples, sub_train_labels
            )
            valid_performance = classifier.evaluate(
                validation_samples, validation_labels
            )
            test_performance = classifier.evaluate(test_samples, test_labels)
            for performance, subset in [
                (sub_train_performance, "subtrain"),
                (valid_performance, "validation"),
                (test_performance, "test"),
            ]:
                all_performance.append(
                    {
                        **performance,
                        "holdout": holdout_number,
                        "subset": subset,
                    }
                )

            del classifier
            clear_session()
            gc.collect()

        pd.DataFrame(all_performance).to_csv(
            "last_performance.csv",
            index=False,
        )

        return pd.DataFrame(all_performance)
