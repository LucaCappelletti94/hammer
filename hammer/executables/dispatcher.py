"""Utility to dispatch a command to the appropriate sub-command.

Implementative details
----------------------
We expect the user to run commands of the form:

    hammer <sub-command> <arguments>

This utility will dispatch the command to the appropriate sub-command.
"""

from argparse import ArgumentParser, Namespace
import silence_tensorflow.auto  # pylint: disable=unused-import
from hammer.executables.feature_sets_evaluation import (
    add_feature_sets_evaluation_subcommand,
)
from hammer.executables.visualize_features import add_visualize_features_subcommand
from hammer.executables.dag_coverage import add_dag_coverage_subcommand


def dispatcher():
    """Dispatch the command to the appropriate sub-command."""
    parser: ArgumentParser = ArgumentParser()
    subparsers = parser.add_subparsers()
    add_feature_sets_evaluation_subcommand(subparsers)
    add_visualize_features_subcommand(subparsers)
    add_dag_coverage_subcommand(subparsers)
    args: Namespace = parser.parse_args()
    args.func(args)
