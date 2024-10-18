"""SMILES Extended Connectivity fingerprint feature implementation using scikit-fingerprints."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.secfp import SECFPFingerprint
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import BinaryFeatureInterface


class SMILESExtendedConnectivity(BinaryFeatureInterface):
    """Class defining the SMILES Extended Connectivity fingerprint feature implementation."""

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 1,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialize the SMILES Extended Connectivity fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        self._radius = radius

        self._fingerprint = SECFPFingerprint(
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
        return f"SMILES Extended Connectivity ({self._radius}r, {self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "smiles_extended_connectivity"

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
            "Subgraphs are created around each atom with increasing radius, "
            "starting with just an atom itself. It is then transformed "
            "into a canonical SMILES and hashed."
        )
