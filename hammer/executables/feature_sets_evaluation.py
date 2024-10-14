"""Script to train the model on each feature set to compare their performance."""

from argparse import Namespace
from typing import List
from tqdm.auto import tqdm
import pandas as pd
from hammer.executables.holdouts_evaluation import holdouts_evaluation
from hammer.feature_settings import FeatureSettings
from hammer.executables.argument_parser_utilities import (
    build_features_settings_from_namespace,
    add_model_selection_arguments,
)
from hammer.executables.feature_sets_synergy import save_performance


def add_feature_sets_evaluation_subcommand(sub_parser_action: "SubParsersAction"):
    """Add the feature sets evaluation sub-command to the parser."""
    subparser = sub_parser_action.add_parser(
        "feature-sets-evaluation",
        help="Evaluate the performance of different feature sets.",
    )
    subparser = add_model_selection_arguments(subparser)

    subparser.set_defaults(func=feature_sets_evaluation)


def feature_sets_evaluation(args: Namespace):
    """Evaluate the performance of different feature sets."""
    performance: List[pd.DataFrame] = []

    feature_settings: FeatureSettings = build_features_settings_from_namespace(args)

    if not feature_settings.includes_features():
        feature_settings = FeatureSettings().include_all()

    for feature_class in tqdm(
        feature_settings.iter_features(),
        desc="Evaluating feature sets",
        unit="feature set",
        total=feature_settings.number_of_features(),
        leave=False,
        dynamic_ncols=True,
        disable=not args.verbose,
    ):
        feature_performance: pd.DataFrame = holdouts_evaluation(
            args,
            feature_settings=FeatureSettings.from_feature_class(feature_class),
            training_directory=args.training_directory,
            performance_path=None,
        )

        feature_performance["feature_set"] = feature_class().name()

        performance.append(feature_performance)

    save_performance(performance, args)
