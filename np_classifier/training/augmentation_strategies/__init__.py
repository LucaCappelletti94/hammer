"""Submodule providing Augmentation strategies"""

from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)
from np_classifier.training.augmentation_strategies.stereoisomers import (
    StereoisomersAugmentationStrategy,
)
from np_classifier.training.augmentation_strategies.tautomers import (
    TautomersAugmentationStrategy,
)

__all__ = [
    "AugmentationStrategy",
    "StereoisomersAugmentationStrategy",
    "TautomersAugmentationStrategy"
]
