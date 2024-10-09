"""Submodule defining the interface for the molecular features."""

from typing import Sequence
from abc import ABC, abstractmethod
from rdkit.Chem.rdchem import Mol
import numpy as np


class FeatureInterface(ABC):
    """Interface for a valid feature."""

    @abstractmethod
    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""

    @abstractmethod
    def name(self) -> str:
        """Get the name of the feature."""

    @staticmethod
    @abstractmethod
    def pythonic_name() -> str:
        """Get the name of the feature in a Pythonic format."""

    @abstractmethod
    def size(self) -> int:
        """Get the size of the feature."""

    @staticmethod
    @abstractmethod
    def low_cardinality() -> bool:
        """Returns whether the feature is defined as low cardinality."""

    @staticmethod
    @abstractmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""

    @staticmethod
    @abstractmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""

    @staticmethod
    @abstractmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
