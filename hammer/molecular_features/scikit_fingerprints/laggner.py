"""Laggner fingerprint feature implementation."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.laggner import LaggnerFingerprint as LaggnerFingerprintSKFP
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.molecular_features.feature_interface import BinaryFeatureInterface


class LaggnerFingerprint(BinaryFeatureInterface):
    """Class defining the Laggner fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Laggner fingerprint feature."""
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()

        self._fingerprint = LaggnerFingerprintSKFP(
            n_jobs=n_jobs,
            verbose={"leave": False, "dynamic_ncols": True, "disable": not verbose},
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "Laggner"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "laggner"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 307
        return self._fingerprint.n_features_out

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "A substructure fingerprint based on SMARTS patterns "
            "for functional group classification, proposed by "
            "Christian Laggner. It tests for presence of 307 "
            "predefined substructures, designed for functional "
            "groups of organic compounds, for use in similarity searching."
        )
