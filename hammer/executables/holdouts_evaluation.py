"""Bash command to train a classifier."""

from typing import Optional
from argparse import Namespace
import silence_tensorflow.auto  # pylint: disable=unused-import
import pandas as pd
import tensorflow as tf
from hammer.training import (
    Trainer,
    Dataset,
    FeatureSettings,
    AugmentationSettings,
)
from hammer.executables.argument_parser_utilities import (
    build_augmentation_settings_from_namespace,
)


def holdouts_evaluation(
    args: Namespace,
    feature_settings: Optional[FeatureSettings] = None,
    performance_path: Optional[str] = "performance.csv",
) -> pd.DataFrame:
    """Train the model."""
    augmentation_settings: AugmentationSettings = (
        build_augmentation_settings_from_namespace(args)
    )

    # We check that a GPU is available, or the training will take a long time
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if not gpu_devices and not args.smoke_test:
        raise RuntimeError("No GPU detected for training, aborting.")

    if feature_settings is None:
        feature_settings = FeatureSettings.standard()

    trainer: Trainer = Trainer(
        maximal_number_of_epochs=1 if args.smoke_test else 10_000,
        smiles_dataset=Dataset(
            maximal_number_of_molecules=2000 if args.smoke_test else None,
            number_of_splits=args.holdouts,
            verbose=args.verbose,
        ),
        feature_settings=feature_settings,
        augmentation_settings=augmentation_settings,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
    )

    performance: pd.DataFrame = trainer.holdouts()
    if performance_path is not None:
        performance.to_csv(performance_path, index=False)
    return performance
