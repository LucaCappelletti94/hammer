"""Submodule populating the argument parser with arguments for model selection."""

from argparse import ArgumentParser


def add_model_selection_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for model selection to the parser."""
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to include in the validation split.",
    )
    return parser
