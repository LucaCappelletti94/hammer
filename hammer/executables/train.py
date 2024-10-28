"""Script to train the model on the selected feature sets."""

from argparse import Namespace
from typing import Dict
from hammer.feature_settings import FeatureSettings
from hammer.training import Trainer
from hammer.augmentation_settings import AugmentationSettings
from hammer.executables.argument_parser_utilities import (
    build_features_settings_from_namespace,
    build_dataset_from_namespace,
    add_model_training_arguments,
    build_augmentation_settings_from_namespace,
)


def add_train_subcommand(sub_parser_action: "SubParsersAction"):
    """Add the feature sets evaluation sub-command to the parser."""
    subparser = sub_parser_action.add_parser(
        "train",
        help="Train the model on the selected feature sets.",
    )
    subparser = add_model_training_arguments(subparser)

    subparser.set_defaults(func=train)


def train(args: Namespace):
    """Evaluate the performance of different feature sets."""
    feature_settings: FeatureSettings = build_features_settings_from_namespace(args)
    augmentation_settings: AugmentationSettings = (
        build_augmentation_settings_from_namespace(args)
    )

    if not feature_settings.includes_features():
        raise RuntimeError("No features selected.")

    trainer: Trainer = Trainer(
        maximal_number_of_epochs=1 if args.smoke_test else 10_000,
        dataset=build_dataset_from_namespace(args),
        feature_settings=feature_settings,
        augmentation_settings=augmentation_settings,
        training_directory=args.training_directory,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
    )

    performance: Dict = trainer.train(test_size=args.test_size)

    print(performance)
