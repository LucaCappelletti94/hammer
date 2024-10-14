"""Submodule implementing a wrapper for the map4 feature."""

from typing import Optional, Sequence
from multiprocessing import cpu_count
from map4 import MAP4 as OriginalMAP4
import numpy as np
from rdkit.Chem.rdchem import Mol
from hammer.features.feature_interface import FeatureInterface


class MAP4(FeatureInterface):
    """Class defining the Autocorrelation fingerprint feature implementation."""

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 1,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialize the Autocorrelation fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._fingerprint = OriginalMAP4(dimensions=fp_size, radius=radius)

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.calculate_many(
            molecules, number_of_threads=self._n_jobs, verbose=self._verbose
        )

    def name(self) -> str:
        """Get the name of the feature."""
        return "MAP4"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "map4"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fingerprint.dimensions

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "It computes fragments based on pairs of atoms, using circular "
            "substructures around each atom represented with SMILES (like SECFP) and length "
            "of shortest path between them (like Atom Pair), and then hashes the resulting "
            "triplet using MinHash fingerprint."
        )
