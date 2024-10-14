"""Submodule providing molecular features wrappers and implementations."""

from hammer.features.feature_interface import FeatureInterface
from hammer.features.map4 import MAP4
from hammer.features.scikit_fingerprints.atom_pair import AtomPairFingerprint
from hammer.features.scikit_fingerprints.autocorrelation import (
    AutocorrelationFingerprint,
)
from hammer.features.scikit_fingerprints.extended_connectivity import (
    ExtendedConnectivityFingerprint,
)
from hammer.features.scikit_fingerprints.functional_groups import (
    FunctionalGroupsFingerprint,
)
from hammer.features.scikit_fingerprints.ghose_crippen import GhoseCrippenFingerprint
from hammer.features.scikit_fingerprints.laggner import LaggnerFingerprint
from hammer.features.scikit_fingerprints.layered import LayeredFingerprint
from hammer.features.scikit_fingerprints.lingo import LingoFingerprint
from hammer.features.scikit_fingerprints.map import MinHashedAtomPairFingerprint
from hammer.features.scikit_fingerprints.minhashed import MinHashedFingerprint
from hammer.features.scikit_fingerprints.molecular_quantum_numbers import (
    MolecularQuantumNumbersFingerprint,
)
from hammer.features.scikit_fingerprints.pattern import PatternFingerprint
from hammer.features.scikit_fingerprints.pubchem import PubChemFingerprint
from hammer.features.scikit_fingerprints.smiles_extended_connectivity import (
    SMILESExtendedConnectivity,
)
from hammer.features.scikit_fingerprints.van_der_waals_surface_area import (
    VanDerWaalsSurfaceAreaFingerprint,
)
from hammer.features.scikit_fingerprints.avalon import AvalonFingerprint
from hammer.features.scikit_fingerprints.maccs import MACCSFingerprint
from hammer.features.scikit_fingerprints.topological_torsion import (
    TopologicalTorsionFingerprint,
)
from hammer.features.scikit_fingerprints.rdkit_fingerprint import RDKitFingerprint

__all__ = [
    "FeatureInterface",
    "AtomPairFingerprint",
    "AutocorrelationFingerprint",
    "ExtendedConnectivityFingerprint",
    "FunctionalGroupsFingerprint",
    "GhoseCrippenFingerprint",
    "LaggnerFingerprint",
    "LayeredFingerprint",
    "LingoFingerprint",
    "MinHashedAtomPairFingerprint",
    "MinHashedFingerprint",
    "MolecularQuantumNumbersFingerprint",
    "PatternFingerprint",
    "PubChemFingerprint",
    "SMILESExtendedConnectivity",
    "VanDerWaalsSurfaceAreaFingerprint",
    "AvalonFingerprint",
    "MACCSFingerprint",
    "TopologicalTorsionFingerprint",
    "RDKitFingerprint",
    "MAP4",
]
