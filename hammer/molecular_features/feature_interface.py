"""Submodule defining the interface for the molecular features."""

from typing import Sequence, Optional
from abc import ABC, abstractmethod
from rdkit.Chem.rdchem import Mol
import numpy as np


class FeatureInterface(ABC):
    """Interface for a valid feature."""

    def __init__(self, n_jobs: Optional[int] = 1, verbose: bool = False) -> None:
        """Initialize the feature."""
        self.n_jobs: int = n_jobs
        self.verbose: bool = verbose

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


class BinaryFeatureInterface(FeatureInterface):
    """Interface for a valid binary feature."""

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8
