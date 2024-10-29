"""Submodule populating the argument parser with arguments for model predictions."""

from argparse import ArgumentParser
from hammer.executables.argument_parser_utilities.shared_arguments import (
    add_shared_arguments,
)


def add_model_predictions_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for model prediction to the parser."""

    # First we set an argument which can be either a single smile,
    # a single InchiKey or a path to a TSV, SSV or CSV file.
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="SMILES, InchiKeys or path to a file (CSV, TSV or SSV) with SMILES or InchiKeys.",
    )

    parser.add_argument(
        "--version",
        type=str,
        required=False,
        help="Path to store the predictions.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to the model to use for the predictions.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Path to store the predictions.",
    )

    parser.add_argument(
        "--only-smiles",
        action="store_true",
        help="Whether to only search for SMILES in the input file.",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        required=False,
        default="csv",
        choices=[
            "csv",
            "tsv",
            "ssv",
            "csv.gz",
            "tsv.gz",
            "ssv.gz",
            "csv.xz",
            "tsv.xz",
            "ssv.xz",
        ],
        help="Format of the output file.",
    )

    parser = add_shared_arguments(parser)

    return parser
