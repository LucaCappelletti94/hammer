"""Submodule providing strategies to extend the smiles in the training dataset."""

from typing import List
from abc import ABC, abstractmethod


class AugmentationStrategy(ABC):
    """Base class for augmentation strategies."""

    @abstractmethod
    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pythonic_name() -> str:
        """Return the pythonic name of the augmentation strategy."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argparse_description() -> str:
        """Return the argparse description of the augmentation strategy."""
        raise NotImplementedError

    @abstractmethod
    def augment(self, smiles: str) -> List[str]:
        """Augment a smiles."""
        raise NotImplementedError

    @abstractmethod
    def augment_all(self, smiles: List[str]) -> List[List[str]]:
        """Augment a list of smiles."""
