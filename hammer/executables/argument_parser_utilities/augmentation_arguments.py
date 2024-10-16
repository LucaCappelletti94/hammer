"""Submodule populating the argument parser with arguments for augmentation settings."""

from argparse import ArgumentParser, Namespace
from hammer.augmentation_settings import STRATEGIES, AugmentationSettings


def add_augmentation_settings_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for augmentation settings to the parser."""
    for strategy_class in STRATEGIES:
        parser.add_argument(
            f"--include-{strategy_class.pythonic_name().replace("_", "-")}",
            type=int,
            default=0,
            help=strategy_class.argparse_description(),
        )
    return parser


def build_augmentation_settings_from_namespace(
    namespace: Namespace,
) -> AugmentationSettings:
    """Build the augmentation settings from the namespace."""
    settings = AugmentationSettings()
    for strategy_class in STRATEGIES:
        if hasattr(namespace, f"include_{strategy_class.pythonic_name()}"):
            value = getattr(namespace, f"include_{strategy_class.pythonic_name()}")
            if value > 0:
                getattr(settings, f"include_{strategy_class.pythonic_name()}")(value)
    return settings
