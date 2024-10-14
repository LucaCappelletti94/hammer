"""Molecular Quantum Numbers fingerprint feature implementation."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.mqns import MQNsFingerprint
from skfp.utils import TQDMSettings
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import FeatureInterface


class MolecularQuantumNumbersFingerprint(FeatureInterface):
    """Class defining the MQNs fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the MQNs fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = MQNsFingerprint(n_jobs=n_jobs, verbose=tqdm_settings)

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "Molecular Quantum Numbers"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "molecular_quantum_numbers"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 42
        return self._fingerprint.n_features_out

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint32

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return False

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Molecular Quantum Numbers fingerprint is a descriptor-based fingerprint, "
            "where bits represent presence of molecular quantum numbers in a molecule."
        )
