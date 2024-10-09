"""Submodule providing wrappers of Scikit Fingerprints."""

from typing import Sequence, Optional
from multiprocessing import cpu_count
from skfp.fingerprints.autocorr import AutocorrFingerprint
from skfp.fingerprints.ecfp import ECFPFingerprint
from skfp.fingerprints.functional_groups import (
    FunctionalGroupsFingerprint as FunctionalGroupsFingerprintSKFP,
)
from skfp.fingerprints.ghose_crippen import (
    GhoseCrippenFingerprint as GhoseCrippenFingerprintSKFP,
)
from skfp.fingerprints.laggner import LaggnerFingerprint as LaggnerFingerprintSKFP
from skfp.fingerprints.layered import LayeredFingerprint as LayeredFingerprintSKFP
from skfp.fingerprints.lingo import LingoFingerprint as LingoFingerprintSKFP
from skfp.fingerprints.map import MAPFingerprint
from skfp.fingerprints.mhfp import MHFPFingerprint
from skfp.fingerprints.mqns import MQNsFingerprint
from skfp.fingerprints.pattern import PatternFingerprint as PatternFingerprintSKFP
from skfp.fingerprints.pubchem import PubChemFingerprint as PubChemFingerprintSKFP
from skfp.fingerprints.secfp import SECFPFingerprint
from skfp.fingerprints.vsa import VSAFingerprint
from skfp.fingerprints.avalon import AvalonFingerprint as AvalonFingerprintSKFP
from skfp.fingerprints.maccs import MACCSFingerprint as MACCSFingerprintSKFP
from skfp.fingerprints.atom_pair import AtomPairFingerprint as AtomPairFingerprintSKFP
from skfp.fingerprints.topological_torsion import (
    TopologicalTorsionFingerprint as TopologicalTorsionFingerprintSKFP,
)
from skfp.fingerprints.rdkit_fp import RDKitFingerprint as RDKitFingerprintSKFP
from skfp.utils import TQDMSettings
from rdkit.Chem.rdchem import Mol
import numpy as np
from hammer.training.features.feature_interface import FeatureInterface


class AutocorrelationFingerprint(FeatureInterface):
    """Class defining the Autocorrelation fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Autocorrelation fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = AutocorrFingerprint(n_jobs=n_jobs, verbose=tqdm_settings)

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "Auto-Correlation"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "autocorrelation"

    def size(self) -> int:
        """Get the size of the feature."""
        return 192

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
            "Autocorrelation fingerprint is descriptor-based fingerprint, "
            "where bits measure strength of autocorrelation of molecular "
            "properties between atoms with different shortest path distances."
        )


class ExtendedConnectivityFingerprint(FeatureInterface):
    """Class defining the Extended Connectivity fingerprint feature implementation."""

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 1,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialize the Extended Connectivity fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        self._radius = radius
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = ECFPFingerprint(
            fp_size=fp_size, radius=radius, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Extended Connectivity ({self._radius}r, {self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "extended_connectivity"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Extended Connectivity fingerprint where fragments "
            "are computed based on circular substructures around each atom."
        )


class FunctionalGroupsFingerprint(FeatureInterface):
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
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Substructure, descriptor fingerprint, checking "
            "occurrences of 85 functional groups."
        )


class GhoseCrippenFingerprint(FeatureInterface):
    """Class defining the Ghose-Crippen fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Ghose-Crippen fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = GhoseCrippenFingerprintSKFP(
            n_jobs=n_jobs, verbose=tqdm_settings
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
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
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


class LaggnerFingerprint(FeatureInterface):
    """Class defining the Laggner fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Laggner fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = LaggnerFingerprintSKFP(n_jobs=n_jobs, verbose=tqdm_settings)

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
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

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


class LayeredFingerprint(FeatureInterface):
    """Class defining the Layered fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 2048, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Layered fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = LayeredFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
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
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a hashed fingerprint, where fragments "
            "are created from small subgraphs on the molecular graph."
        )


class LingoFingerprint(FeatureInterface):
    """Class defining the Lingo fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 1024, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Lingo fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = LingoFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Lingo ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "lingo"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "The Lingo fingerprint is a hashed fingerprint that checks "
            "the occurrences of substrings of a given length in a SMILES string."
        )


class MinHashedAtomPairFingerprint(FeatureInterface):
    """Class defining the MAP fingerprint feature implementation."""

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialize the MAP fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        self._radius = radius
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = MAPFingerprint(
            fp_size=fp_size, radius=radius, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"MinHashed Atom Pair ({self._radius}r, {self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "minhashed_atom_pair"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a hashed fingerprint, using the ideas "
            "from Atom Pair and SECFP fingerprints."
        )


class MinHashedFingerprint(FeatureInterface):
    """Class defining the MHFP fingerprint feature implementation."""

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialize the MHFP fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        self._radius = radius
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = MHFPFingerprint(
            fp_size=fp_size, radius=radius, n_jobs=n_jobs, verbose=tqdm_settings
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
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Hash where fragments are computed based "
            "on circular substructures around each atom."
        )


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


class PatternFingerprint(FeatureInterface):
    """Class defining the Pattern fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 2048, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Pattern fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = PatternFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Pattern ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "pattern"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Pattern fingerprint is a descriptor-based fingerprint, "
            "where bits represent presence of patterns in a molecule."
        )


class PubChemFingerprint(FeatureInterface):
    """Class defining the PubChem fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the PubChem fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = PubChemFingerprintSKFP(n_jobs=n_jobs, verbose=tqdm_settings)

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "PubChem"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "pubchem"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 881
        return self._fingerprint.n_features_out

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return "PubChem fingerprint, also known as CACTVS fingerprint."


class SMILESExtendedConnectivity(FeatureInterface):
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
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = SECFPFingerprint(
            fp_size=fp_size, radius=radius, n_jobs=n_jobs, verbose=tqdm_settings
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
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "Subgraphs are created around each atom with increasing radius, "
            "starting with just an atom itself. It is then transformed "
            "into a canonical SMILES and hashed."
        )


class VanDerWaalsSurfaceAreaFingerprint(FeatureInterface):
    """Class defining the Van Der Waals Surface Area fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the Van Der Waals Surface Area fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = VSAFingerprint(
            variant="all", n_jobs=n_jobs, verbose=tqdm_settings
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


class AvalonFingerprint(FeatureInterface):
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
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return "Avalon fingerprint"


class MACCSFingerprint(FeatureInterface):
    """Class defining the MACCS fingerprint feature implementation."""

    def __init__(self, verbose: bool = True, n_jobs: Optional[int] = None) -> None:
        """Initialize the MACCS fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = MACCSFingerprintSKFP(n_jobs=n_jobs, verbose=tqdm_settings)

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return "MACCS"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "maccs"

    def size(self) -> int:
        """Get the size of the feature."""
        # Features equal to 166
        return self._fingerprint.n_features_out

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return True

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a substructure fingerprint, based "
            "on publicly available MDL definitions, and "
            "refined by Gregory Landrum for RDKit."
        )


class AtomPairFingerprint(FeatureInterface):
    """Class defining the Atom Pair fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 2048, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Atom Pair fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = AtomPairFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"Atom Pair ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "atom_pair"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a hashed fingerprint, where fragments "
            "are computed based on pairs of atoms and distance between them."
        )


class TopologicalTorsionFingerprint(FeatureInterface):
    """Class defining the Topological Torsion fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 1024, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the Topological Torsion fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = TopologicalTorsionFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
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
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a hashed fingerprint, where the hashed "
            "fragments are computed based on topological torsions"
        )


class RDKitFingerprint(FeatureInterface):
    """Class defining the RDKit fingerprint feature implementation."""

    def __init__(
        self, fp_size: int = 2048, verbose: bool = True, n_jobs: Optional[int] = None
    ) -> None:
        """Initialize the RDKit fingerprint feature."""
        if n_jobs is None or n_jobs < 1:
            n_jobs = cpu_count()
        self._fp_size = fp_size
        tqdm_settings = (
            TQDMSettings().leave(False).desc(self.name()).dynamic_ncols(True)
        )
        if not verbose:
            tqdm_settings = tqdm_settings.disable()
        self._fingerprint = RDKitFingerprintSKFP(
            fp_size=fp_size, n_jobs=n_jobs, verbose=tqdm_settings
        )

    def transform_molecules(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Transform a molecule into a feature representation."""
        return self._fingerprint.transform(molecules)

    def name(self) -> str:
        """Get the name of the feature."""
        return f"RDKit ({self._fp_size}b)"

    @staticmethod
    def pythonic_name() -> str:
        """Get the pythonic name of the feature."""
        return "rdkit"

    def size(self) -> int:
        """Get the size of the feature."""
        return self._fp_size

    @staticmethod
    def low_cardinality() -> bool:
        """Return whether the feature has low cardinality."""
        return False

    @staticmethod
    def dtype() -> np.dtype:
        """Get the data type of the feature."""
        return np.uint8

    @staticmethod
    def is_binary() -> bool:
        """Returns whether the feature is binary."""
        return True

    @staticmethod
    def argparse_description() -> str:
        """Get the argparse description of the feature."""
        return (
            "This is a hashed fingerprint, where fragments "
            "are created from small subgraphs on the molecular graph."
        )
