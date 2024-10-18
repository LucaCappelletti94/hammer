"""Implementation of the Layered fingerprint feature."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.layered import LayeredFingerprint as LayeredFingerprintSKFP
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import BinaryFeatureInterface


class LayeredFingerprint(BinaryFeatureInterface):
    """Class defining the Layered fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 2048, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Layered fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size

        self._fingerprint = LayeredFingerprintSKFP(
            fp_size=fp_size,
            n_jobs=n_jobs,
            verbose={"leave": False, "dynamic_ncols": True, "disable": not verbose},
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Layered ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "layered"

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
            "This is a hashed fingerprint, where fragments "
            "are created from small subgraphs on the molecular graph."
        )
