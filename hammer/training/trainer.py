"""Submodule handling the training of the multi-modal multi-class classifier."""

import os
from typing import Optional
import pandas as pd
from hammer.training.smiles_dataset import Dataset
from hammer.training.model import Classifier
from hammer.training.feature_settings import FeatureSettings
from hammer.training.augmentation_settings import AugmentationSettings


class Trainer:
    """Class handling the training of the multi-modal multi-class classifier."""

    def __init__(
        self,
        smiles_dataset: Dataset,
        feature_settings: FeatureSettings,
        augmentation_settings: AugmentationSettings,
        maximal_number_of_epochs: int = 10_000,
        model_directory: Optional[str] = None,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ):
        """Initialize the trainer."""
        assert isinstance(smiles_dataset, Dataset)
        self._smiles_dataset = smiles_dataset
        self._maximal_number_of_epochs = maximal_number_of_epochs
        self._feature_settings = feature_settings
        self._augmentation_settings = augmentation_settings
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._model_directory = model_directory

    def holdouts(
        self,
    ) -> pd.DataFrame:
        """Train the classifier."""
        (_train_smiles, _train_labels), (test_smiles, test_labels) = (
            self._smiles_dataset.primary_split()
        )
        all_performance = []
        for holdout_number, (
            (sub_train_smiles, sub_train_labels),
            (validation_smiles, validation_labels),
        ) in enumerate(self._smiles_dataset.train_split()):
            classifier = Classifier(
                class_names=self._smiles_dataset.class_names,
                superclass_names=self._smiles_dataset.superclass_names,
                pathway_names=self._smiles_dataset.pathway_names,
                feature_settings=self._feature_settings,
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
            if self._model_directory is not None:
                classifier.save(
                    os.path.join(
                        self._model_directory, f"holdout_{holdout_number}.tar.gz"
                    )
                )
            sub_train_performance = classifier.evaluate(
                sub_train_smiles, sub_train_labels
            )
            sub_valid_performance = classifier.evaluate(
                validation_smiles, validation_labels
            )
            test_performance = classifier.evaluate(test_smiles, test_labels)
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
        return pd.DataFrame(all_performance)
