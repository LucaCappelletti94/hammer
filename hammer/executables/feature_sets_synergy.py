"""Train the model on one feature sets plus others to determine their synergy."""

import os
from argparse import Namespace
from typing import List
from tqdm.auto import tqdm
import pandas as pd
from barplots import barplots
from hammer.executables.holdouts_evaluation import holdouts_evaluation
from hammer.feature_settings import FeatureSettings, FEATURES
from hammer.executables.argument_parser_utilities import (
    add_model_selection_arguments,
)


def add_feature_sets_synergy_subcommand(sub_parser_action: "SubParsersAction"):
    """Add the feature sets synergy sub-command to the parser."""
    subparser = sub_parser_action.add_parser(
        "feature-sets-synergy",
        help="Evaluate the performance of different feature sets.",
    )
    subparser = add_model_selection_arguments(subparser)

    subparser.add_argument(
        "--base-feature-sets",
        type=str,
        nargs="+",
        required=True,
        help="Feature sets to use as base.",
    )

    subparser.set_defaults(func=feature_sets_synergy)


def save_performance(performance: List[pd.DataFrame], args: Namespace):
    """Save the performance of the feature sets."""
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


def feature_sets_synergy(args: Namespace):
    """Evaluate the performance of different feature sets."""
    performance: List[pd.DataFrame] = []

    feature_settings: FeatureSettings = FeatureSettings()
    low_cardinality_features: FeatureSettings = FeatureSettings().include_all()

    for feature_class in FEATURES:
        if feature_class.pythonic_name() in args.base_feature_sets:
            getattr(feature_settings, f"include_{feature_class.pythonic_name()}")()
            getattr(
                low_cardinality_features, f"remove_{feature_class.pythonic_name()}"
            )()

    for feature_class in tqdm(
        low_cardinality_features.iter_features(),
        desc="Evaluating feature sets synergy",
        unit="feature set",
        total=low_cardinality_features.number_of_features(),
        leave=False,
        dynamic_ncols=True,
        disable=not args.verbose,
    ):
        feature_settings_copy = feature_settings.copy()
        getattr(feature_settings_copy, f"include_{feature_class.pythonic_name()}")()

        assert (
            feature_settings_copy.number_of_features()
            == feature_settings.number_of_features() + 1
        )

        feature_performance: pd.DataFrame = holdouts_evaluation(
            args,
            feature_settings=feature_settings_copy,
            training_directory=args.training_directory,
            performance_path=None,
        )

        feature_performance["feature_set"] = feature_class().name()

        performance.append(feature_performance)

        save_performance(performance, args)

    save_performance(performance, args)
