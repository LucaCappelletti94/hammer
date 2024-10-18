"""Ghose-Crippen fingerprint feature implementation."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.ghose_crippen import (
    GhoseCrippenFingerprint as GhoseCrippenFingerprintSKFP,
)
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import BinaryFeatureInterface


class GhoseCrippenFingerprint(BinaryFeatureInterface):
    """Class defining the Ghose-Crippen fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Ghose-Crippen fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()

        self._fingerprint = GhoseCrippenFingerprintSKFP(
            n_jobs=n_jobs,
            verbose={"leave": False, "dynamic_ncols": True, "disable": not verbose},
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "Ghose-Crippen"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "ghose_crippen"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 110
        return self._fingerprint.n_features_out

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "A substructure fingerprint based on 110 atom types "
            "proposed by Ghose and Crippen. They are defined for "
            "carbon, hydrogen, oxygen, nitrogen, sulfur, and halogens, "
            "and originally applied for predicting molar refractivities and logP."
        )
