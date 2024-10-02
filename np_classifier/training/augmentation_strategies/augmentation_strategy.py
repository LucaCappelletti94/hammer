"""Submodule providing strategies to extend the smiles in the training dataset."""

from typing import List
from abc import ABC, abstractmethod


class AugmentationStrategy(ABC):
    """Base class for augmentation strategies."""

    @abstractmethod
    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        raise NotImplementedError

    @abstractmethod
    def augment(self, smiles: str) -> List[str]:
        """Augment a smiles."""
        raise NotImplementedError
