"""Avalon fingerprint feature implementation using scikit-fingerprints."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.avalon import AvalonFingerprint as AvalonFingerprintSKFP
from skfp.utils import TQDMSettings
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import BinaryFeatureInterface


class AvalonFingerprint(BinaryFeatureInterface):
    """Class defining the Avalon fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 2048, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Avalon fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = AvalonFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Avalon ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "avalon"

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
        return "Avalon fingerprint"
