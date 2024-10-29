"""Utility to dispatch a command to the appropriate sub-command.

Implementative details
----------------------
We expect the user to run commands of the form:

    hammer <sub-command> <arguments>

This utility will dispatch the command to the appropriate sub-command.
"""

from argparse import ArgumentParser, Namespace
from hammer.executables.feature_sets_evaluation import (
    add_feature_sets_evaluation_subcommand,
)
from hammer.executables.visualize_features import add_visualize_features_subcommand
from hammer.executables.dag_coverage import add_dag_coverage_subcommand
from hammer.executables.feature_sets_synergy import add_feature_sets_synergy_subcommand
from hammer.executables.train import add_train_subcommand
from hammer.executables.predict import add_predict_subcommand
from hammer.executables.holdouts_evaluation import add_holdouts_evaluation_subcommand


def dispatcher() -> None:
    """Dispatch the command to the appropriate sub-command."""
    parser: ArgumentParser = ArgumentParser()
    subparsers = parser.add_subparsers()
    add_holdouts_evaluation_subcommand(
        subparsers.add_parser(
            "holdouts",
            help="Evaluate the model with the selected feature set.",
        )
    )
    add_feature_sets_evaluation_subcommand(
        subparsers.add_parser(
            "feature-sets-evaluation",
            help="Evaluate the performance of different feature sets.",
        )
    )
    add_visualize_features_subcommand(
        subparsers.add_parser(
            "visualize", help="Visualize the features of the dataset."
        )
    )
    add_feature_sets_synergy_subcommand(
        subparsers.add_parser(
            "feature-sets-synergy",
            help="Evaluate the performance of different feature sets.",
        )
    )
    add_dag_coverage_subcommand(
        subparsers.add_parser(
            "dag-coverage",
            help="Compute the dataset coverage of the current chemical DAG.",
        )
    )
    add_train_subcommand(
        subparsers.add_parser(
            "train",
            help="Train the model on the selected feature sets.",
        )
    )
    add_predict_subcommand(
        subparsers.add_parser(
            "predict",
            help="Run model predictions.",
        )
    )
    args: Namespace = parser.parse_args()
    args.func(args)
