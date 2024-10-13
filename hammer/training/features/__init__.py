"""Submodule providing molecular features wrappers and implementations."""

from hammer.training.features.feature_interface import FeatureInterface
from hammer.training.features.map4 import MAP4
from hammer.training.features.scikit_fingerprints import (
    AtomPairFingerprint,
    AutocorrelationFingerprint,
    ExtendedConnectivityFingerprint,
    FunctionalGroupsFingerprint,
    GhoseCrippenFingerprint,
    LaggnerFingerprint,
    LayeredFingerprint,
    LingoFingerprint,
    MinHashedAtomPairFingerprint,
    MinHashedFingerprint,
    MolecularQuantumNumbersFingerprint,
    PatternFingerprint,
    PubChemFingerprint,
    SMILESExtendedConnectivity,
    VanDerWaalsSurfaceAreaFingerprint,
    AvalonFingerprint,
    MACCSFingerprint,
    TopologicalTorsionFingerprint,
    RDKitFingerprint,
)

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
