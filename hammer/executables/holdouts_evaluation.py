"""Bash command to train a classifier."""

from typing import Optional
from argparse import Namespace
import pandas as pd
from environments_utils import has_nvidia_gpu, has_amd_gpu
from hammer.training import (
    Trainer,
)
from hammer.feature_settings import (
    FeatureSettings,
)
from hammer.augmentation_settings import (
    AugmentationSettings,
)
from hammer.executables.argument_parser_utilities import (
    build_augmentation_settings_from_namespace,
    build_dataset_from_namespace,
)


def holdouts_evaluation(
    args: Namespace,
    feature_settings: Optional[FeatureSettings] = None,
    training_directory: str = "trained_models",
    performance_path: Optional[str] = "performance.csv",
) -> pd.DataFrame:
    """Train the model."""
    augmentation_settings: AugmentationSettings = (
        build_augmentation_settings_from_namespace(args)
    )

    # We check that a GPU is available, or the training will take a long time
    if not (has_nvidia_gpu() or has_amd_gpu()) and not args.smoke_test:
        raise RuntimeError("No GPU detected for training, aborting.")

    if feature_settings is None:
        feature_settings = FeatureSettings.standard()

    trainer: Trainer = Trainer(
        maximal_number_of_epochs=1 if args.smoke_test else 10_000,
        dataset=build_dataset_from_namespace(args),
        feature_settings=feature_settings,
        augmentation_settings=augmentation_settings,
        training_directory=training_directory,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
    )

    performance: pd.DataFrame = trainer.holdouts(
        number_of_holdouts=args.holdouts,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )
    if performance_path is not None:
        performance.to_csv(performance_path, index=False)
    return performance
