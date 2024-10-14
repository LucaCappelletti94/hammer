"""Functional Groups fingerprint feature implementation."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.functional_groups import (
    FunctionalGroupsFingerprint as FunctionalGroupsFingerprintSKFP,
)
from skfp.utils import TQDMSettings
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import BinaryFeatureInterface


class FunctionalGroupsFingerprint(BinaryFeatureInterface):
    """Class defining the Functional Groups fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Functional Groups fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = FunctionalGroupsFingerprintSKFP(
            n_jobs=n_jobs, verbose=tqdm_settings
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
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Substructure, descriptor fingerprint, checking "
            "occurrences of 85 functional groups."
        )