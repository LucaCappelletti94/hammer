"""Submodule for SMILES dataclass."""

from typing import List, Dict
import numpy as np
from rdkit.Chem import Mol # pylint: disable=no-name-in-module
from rdkit.Chem import MolFromSmiles # pylint: disable=no-name-in-module
from np_classifier.utils import is_glycoside
from np_classifier.utils.constants import PATHWAY_NAMES, SUPERCLASS_NAMES, CLASS_NAMES
from np_classifier.utils import to_morgan_fingerprint


class Molecule:
    """Dataclass representing a Molecule with its label."""

    molecule: Mol
    pathway_labels: np.ndarray
    superclass_labels: np.ndarray
    class_labels: np.ndarray

    def __init__(
        self,
        molecule: Mol,
        pathway_labels: np.ndarray,
        superclass_labels: np.ndarray,
        class_labels: np.ndarray,
    ):
        """Initialize the SMILES dataclass."""
        assert isinstance(molecule, Mol)
        assert isinstance(pathway_labels, np.ndarray)
        assert isinstance(superclass_labels, np.ndarray)
        assert isinstance(class_labels, np.ndarray)
        assert pathway_labels.dtype == np.int32
        assert superclass_labels.dtype == np.int32
        assert class_labels.dtype == np.int32
        assert pathway_labels.size > 0
        assert superclass_labels.size > 0
        assert class_labels.size > 0
        self.molecule = molecule
        self.pathway_labels = pathway_labels
        self.superclass_labels = superclass_labels
        self.class_labels = class_labels

    @staticmethod
    def from_smiles(
        smiles: str,
        pathway_labels: List[str],
        superclass_labels: List[str],
        class_labels: List[str],
    ) -> "Molecule":
        """Create a Molecule from a SMILES string."""
        return Molecule(
            molecule=MolFromSmiles(smiles),
            pathway_labels=np.fromiter(
                (PATHWAY_NAMES.index(label) for label in pathway_labels), dtype=np.int32
            ),
            superclass_labels=np.fromiter(
                (SUPERCLASS_NAMES.index(label) for label in superclass_labels),
                dtype=np.int32,
            ),
            class_labels=np.fromiter(
                (CLASS_NAMES.index(label) for label in class_labels), dtype=np.int32
            ),
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
        return is_glycoside(self.molecule)

    def into_homologue(self, homologue: Mol) -> "Smiles":
        """Return a homologue of the molecule."""
        return Molecule(
            molecule=homologue,
            pathway_labels=self.pathway_labels,
            superclass_labels=self.superclass_labels,
            class_labels=self.class_labels,
        )

    def fingerprint(self, radius: int = 3, n_bits: int = 2048) -> Dict[str, np.ndarray]:
        """Return the Morgan fingerprint of the molecule."""
        formula_fingerprint, binary_fingerprint = to_morgan_fingerprint(
            self.molecule, radius=radius, n_bits=n_bits
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