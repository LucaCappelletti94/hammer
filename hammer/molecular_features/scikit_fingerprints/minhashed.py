"""MinHashed fingerprint feature implementation."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.mhfp import MHFPFingerprint
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.molecular_features.feature_interface import BinaryFeatureInterface


class MinHashedFingerprint(BinaryFeatureInterface):
    """Class defining the MHFP fingerprint feature implementation."""

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialize the MHFP fingerprint feature."""
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        self._radius = radius

        self._fingerprint = MHFPFingerprint(
            fp_size=fp_size,
            radius=radius,
            n_jobs=n_jobs,
            verbose={"leave": False, "dynamic_ncols": True, "disable": not verbose},
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"MinHashed ({self._radius}r, {self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "minhashed"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Hash where fragments are computed based "
            "on circular substructures around each atom."
        )
