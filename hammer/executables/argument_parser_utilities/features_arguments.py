"""Submodule populating the argument parser with arguments for feature settings."""

from argparse import ArgumentParser, Namespace
from hammer.feature_settings import FeatureSettings, FEATURES


def add_features_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for feature settings to the parser."""
    for feature_class in FEATURES:
        parser.add_argument(
            f"--include-{feature_class.pythonic_name().replace("_", "-")}",
            action="store_true",
            default=False,
            help=feature_class.argparse_description(),
        )
    return parser


def build_features_settings_from_namespace(namespace: Namespace) -> FeatureSettings:
    """Build the feature settings from the namespace."""
    settings = FeatureSettings()
    for feature_class in FEATURES:
        if hasattr(namespace, f"include_{feature_class.pythonic_name()}"):
            value = getattr(namespace, f"include_{feature_class.pythonic_name()}")
            if value:
                getattr(settings, f"include_{feature_class.pythonic_name()}")()
    return settings
