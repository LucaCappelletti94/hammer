"""Script to train the model on each feature set to compare their performance."""

import os
from argparse import Namespace
from typing import List
from tqdm.auto import tqdm
import pandas as pd
from barplots import barplots
from hammer.executables.holdouts_evaluation import holdouts_evaluation
from hammer.training import FeatureSettings
from hammer.training.feature_settings import FEATURES
from hammer.executables.argument_parser_utilities import (
    add_shared_arguments,
    add_augmentation_settings_arguments,
)


def add_feature_sets_evaluation_subcommand(sub_parser_action: "SubParsersAction"):
    """Add the feature sets evaluation sub-command to the parser."""
    subparser = sub_parser_action.add_parser(
        "feature-sets-evaluation",
        help="Evaluate the performance of different feature sets.",
    )
    subparser = add_augmentation_settings_arguments(add_shared_arguments(subparser))

    # Adds an argument for where to store the performance
    subparser.add_argument(
        "--performance-path",
        type=str,
        default="feature_sets_evaluation.csv",
        help="Path to store the performance of the feature sets.",
    )

    # Adds an argument for where to store the barplots of the performance
    subparser.add_argument(
        "--barplot-directory",
        type=str,
        default="feature_sets_evaluation_barplots",
        help="Path to store the barplots of the performance.",
    )

    # Adds an argument for the number of holdouts to execute.
    subparser.add_argument(
        "--holdouts",
        type=int,
        default=10,
        help="Number of holdouts to execute.",
    )

    subparser.set_defaults(func=feature_sets_evaluation)


def feature_sets_evaluation(args: Namespace):
    """Evaluate the performance of different feature sets."""
    performance: List[pd.DataFrame] = []

    for feature_class in tqdm(
        FEATURES,
        desc="Evaluating feature sets",
        unit="feature set",
        leave=False,
        dynamic_ncols=True,
        disable=not args.verbose,
    ):
        feature_performance: pd.DataFrame = holdouts_evaluation(
            args,
            feature_settings=FeatureSettings.from_feature_class(feature_class),
            performance_path=None,
        )

        feature_performance["feature_set"] = feature_class().name()

        performance.append(feature_performance)

    performance_df = pd.concat(performance)
    performance_directory = os.path.dirname(args.performance_path)
    if performance_directory:
        os.makedirs(performance_directory, exist_ok=True)
    performance_df.to_csv(args.performance_path, index=False)

    performance_df = performance_df.drop(columns=["holdout"])

    barplots(
        performance_df,
        path=f"{args.barplot_directory}/{{feature}}_feature_sets.png",
        groupby=["subset", "feature_set"],
        show_last_level_as_legend=False,
        subplots=True,
        unique_minor_labels=False,
        orientation="horizontal",
        height=7,
        legend_position="lower left",
    )
