"""Submodule providing strategies to extend the smiles in the training dataset."""

from typing import List
from abc import ABC, abstractmethod
from np_classifier.training.molecule import Molecule


class AugmentationStrategy(ABC):
    """Base class for augmentation strategies."""

    @abstractmethod
    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        raise NotImplementedError

    @abstractmethod
    def augment(self, molecule: Molecule) -> List[Molecule]:
        """Augment a molecule."""
        raise NotImplementedError
