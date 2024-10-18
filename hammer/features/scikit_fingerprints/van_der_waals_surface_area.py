"""Van Der Waals Surface Area fingerprint feature implementation using scikit-fingerprints."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.vsa import VSAFingerprint
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.features.feature_interface import FeatureInterface


class VanDerWaalsSurfaceAreaFingerprint(FeatureInterface):
    """Class defining the Van Der Waals Surface Area fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Van Der Waals Surface Area fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        
        self._fingerprint = VSAFingerprint(
            variant="all", n_jobs=n_jobs, verbose={
                "leave": False,
                "dynamic_ncols": True,
                "disable": not verbose
            }
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "Van Der Waals Surface Area"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "van_der_waals_surface_area"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 47
        return self._fingerprint.n_features_out

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.float32

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return False

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "To calculate VSA, one gets the contribution of each atom "
            "in the molecule to a molecular property (e.g. SLogP) along "
            "with the contribution of each atom to the approximate molecular "
            "surface area (VSA), assign the atoms to bins based on the "
            "property contributions, and then sum up the VSA contributions "
            "for each atom in a bin."
        )
