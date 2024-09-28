"""Submodule for SMILES dataclass."""

from typing import List, Dict
import numpy as np
from np_classifier.utils import is_glycoside
from np_classifier.utils.constants import PATHWAY_NAMES, SUPERCLASS_NAMES, CLASS_NAMES
from np_classifier.utils import smiles_to_morgan_fingerprint


class Smiles:
    """Dataclass representing a SMILES string with its label."""

    smiles: str
    pathway_labels: np.ndarray
    superclass_labels: np.ndarray
    class_labels: np.ndarray

    def __init__(
        self,
        smiles: str,
        pathway_labels: List[str],
        superclass_labels: List[str],
        class_labels: List[str],
    ):
        """Initialize the SMILES dataclass."""
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        assert isinstance(pathway_labels, list)
        assert len(pathway_labels) > 0
        assert isinstance(superclass_labels, list)
        assert len(superclass_labels) > 0
        assert isinstance(class_labels, list)
        assert len(class_labels) > 0

        self.smiles = smiles
        self.pathway_labels = np.fromiter(
            (PATHWAY_NAMES.index(label) for label in pathway_labels), dtype=np.int32
        )
        self.superclass_labels = np.fromiter(
            (SUPERCLASS_NAMES.index(label) for label in superclass_labels),
            dtype=np.int32,
        )
        self.class_labels = np.fromiter(
            (CLASS_NAMES.index(label) for label in class_labels), dtype=np.int32
        )

    @property
    def one_hot_pathway(self) -> np.ndarray:
        """Return the one-hot encoded pathway label."""
        one_hot = np.zeros(len(PATHWAY_NAMES), dtype=np.float32)
        one_hot[self.pathway_labels] = 1
        return one_hot

    @property
    def one_hot_superclass(self) -> np.ndarray:
        """Return the one-hot encoded superclass label."""
        one_hot = np.zeros(len(SUPERCLASS_NAMES), dtype=np.float32)
        one_hot[self.superclass_labels] = 1
        return one_hot

    @property
    def one_hot_class(self) -> np.ndarray:
        """Return the one-hot encoded class label."""
        one_hot = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        one_hot[self.class_labels] = 1
        return one_hot

    @property
    def first_class_label(self) -> int:
        """Return the first class label."""
        return self.class_labels[0]

    def is_glycoside(self) -> bool:
        """Return whether the molecule is a glycoside."""
        return is_glycoside(self.smiles)

    def into_homologue(self, homologue: str) -> "Smiles":
        """Return a homologue of the molecule."""
        return Smiles(
            smiles=homologue,
            pathway_labels=[PATHWAY_NAMES[label] for label in self.pathway_labels],
            superclass_labels=[
                SUPERCLASS_NAMES[label] for label in self.superclass_labels
            ],
            class_labels=[CLASS_NAMES[label] for label in self.class_labels],
        )

    def fingerprint(self, radius: int = 3, n_bits: int = 2048) -> Dict[str, np.ndarray]:
        """Return the Morgan fingerprint of the molecule."""
        formula_fingerprint, binary_fingerprint = smiles_to_morgan_fingerprint(
            self.smiles, radius=radius, n_bits=n_bits
        )
        return {
            "binary_fingerprint": binary_fingerprint,
            "formula_fingerprint": formula_fingerprint,
        }

    def labels(self) -> Dict[str, np.ndarray]:
        """Return the labels."""
        return {
            "class": self.one_hot_class,
            "pathway": self.one_hot_pathway,
            "superclass": self.one_hot_superclass,
        }