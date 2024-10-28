"""Script to compute the dataset coverage of the current chemical DAG."""

from argparse import Namespace, ArgumentParser
from typing import Dict, Type
import pandas as pd
from hammer.datasets import Dataset
from hammer.executables.argument_parser_utilities import (
    add_shared_arguments,
    add_dataset_arguments,
    build_dataset_from_namespace,
)


def add_dag_coverage_subcommand(subparser: ArgumentParser):
    """Add the DAG coverage sub-command to the parser."""
    subparser = add_dataset_arguments(add_shared_arguments(subparser))

    subparser.set_defaults(func=compute_dag_coverage)


def compute_dag_coverage(args: Namespace) -> None:
    """Compute the dataset coverage of the current chemical DAG."""
    dataset: Type[Dataset] = build_dataset_from_namespace(namespace=args)
    dag_coverage: float = dataset.dag_coverage()
    layer_coverages: Dict[str, float] = dataset.dag_layer_coverage()

    dataframe: pd.DataFrame = pd.DataFrame(
        {
            "Dataset": [args.dataset for _ in range(len(layer_coverages) + 1)],
            "Layer": list(layer_coverages.keys()) + ["DAG"],
            "Coverage": list(layer_coverages.values()) + [dag_coverage],
        }
    )

    print(dataframe.to_markdown(index=False))
