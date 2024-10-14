"""Submodule defining argparse arguments relative to dataset selection and parametrization."""

from argparse import ArgumentParser, Namespace
from hammer.datasets import AVAILABLE_DATASETS


def add_dataset_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for dataset selection to the parser."""
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[dataset.name() for dataset in AVAILABLE_DATASETS],
        required=True,
        help="The dataset to use for training.",
    )
    return parser


def build_dataset_from_namespace(namespace: Namespace) -> AVAILABLE_DATASETS:
    """Build the dataset from the namespace."""
    dataset_name = getattr(namespace, "dataset")
    for dataset_class in AVAILABLE_DATASETS:
        if dataset_class.name() == dataset_name:
            return dataset_class(
                random_state=namespace.random_state,
                maximal_number_of_molecules=2000 if namespace.smoke_test else None,
                verbose=namespace.verbose,
            )
    raise ValueError(f"Unknown dataset {dataset_name}")
