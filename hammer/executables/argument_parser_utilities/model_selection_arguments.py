"""Submodule populating the argument parser with arguments for model selection."""

from argparse import ArgumentParser
from hammer.executables.argument_parser_utilities.augmentation_arguments import (
    add_augmentation_settings_arguments,
)
from hammer.executables.argument_parser_utilities.dataset_arguments import (
    add_dataset_arguments,
)
from hammer.executables.argument_parser_utilities.features_arguments import (
    add_features_arguments,
)
from hammer.executables.argument_parser_utilities.shared_arguments import (
    add_shared_arguments,
)


def add_model_training_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for model selection to the parser."""
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to include in the test split.",
    )

    # Adds an argument for where to store the models being trained.
    parser.add_argument(
        "--training-directory",
        type=str,
        default="trained_models",
        help="Path to store the trained models.",
    )

    parser = add_dataset_arguments(
        add_features_arguments(
            add_augmentation_settings_arguments(add_shared_arguments(parser))
        )
    )

    return parser


def add_model_selection_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for model selection to the parser."""

    parser = add_model_training_arguments(parser)

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to include in the validation split.",
    )

    # Adds an argument for the number of holdouts to execute.
    parser.add_argument(
        "--holdouts",
        type=int,
        default=10,
        help="Number of holdouts to execute.",
    )

    # Adds an argument for where to store the performance
    parser.add_argument(
        "--performance-path",
        type=str,
        default="feature_sets_evaluation.csv",
        help="Path to store the performance of the feature sets.",
    )

    # Adds an argument for where to store the barplots of the performance
    parser.add_argument(
        "--barplot-directory",
        type=str,
        default="feature_sets_evaluation_barplots",
        help="Path to store the barplots of the performance.",
    )

    return parser
