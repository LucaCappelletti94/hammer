"""Topological Torsion fingerprint feature implementation using scikit-fingerprints."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.topological_torsion import (
    TopologicalTorsionFingerprint as TopologicalTorsionFingerprintSKFP,
)
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import BinaryFeatureInterface


class TopologicalTorsionFingerprint(BinaryFeatureInterface):
    """Class defining the Topological Torsion fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 1024, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Topological Torsion fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size

        self._fingerprint = TopologicalTorsionFingerprintSKFP(
            fp_size=fp_size,
            n_jobs=n_jobs,
            verbose={"leave": False, "dynamic_ncols": True, "disable": not verbose},
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Topological Torsion ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "topological_torsion"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a hashed fingerprint, where the hashed "
            "fragments are computed based on topological torsions"
        )
