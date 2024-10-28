"""Functional Groups fingerprint feature implementation."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.functional_groups import (
    FunctionalGroupsFingerprint as FunctionalGroupsFingerprintSKFP,
)
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.molecular_features.feature_interface import BinaryFeatureInterface


class FunctionalGroupsFingerprint(BinaryFeatureInterface):
    """Class defining the Functional Groups fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Functional Groups fingerprint feature."""
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()

        self._fingerprint = FunctionalGroupsFingerprintSKFP(
            n_jobs=n_jobs,
            verbose={"leave": False, "dynamic_ncols": True, "disable": not verbose},
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "Functional Groups"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "functional_groups"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 85
        return self._fingerprint.n_features_out

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Substructure, descriptor fingerprint, checking "
            "occurrences of 85 functional groups."
        )
