"""Submodule for SMILES dataclass."""

from typing import List, Dict
import numpy as np
from rdkit.Chem import Mol  # pylint: disable=no-name-in-module
from rdkit.Chem import MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem import MolFromSmarts  # pylint: disable=no-name-in-module
from rdkit.Chem import rdFingerprintGenerator  # pylint: disable=no-name-in-module
from rdkit.Chem import AddHs  # pylint: disable=no-name-in-module
from rdkit.Chem import MACCSkeys  # pylint: disable=no-name-in-module
from rdkit.Chem import Descriptors  # pylint: disable=no-name-in-module
from rdkit.Chem import Lipinski  # pylint: disable=no-name-in-module
from rdkit.Chem import Crippen  # pylint: disable=no-name-in-module
from rdkit.Chem import rdMolDescriptors # pylint: disable=no-name-in-module

from np_classifier.utils.constants import PATHWAY_NAMES, SUPERCLASS_NAMES, CLASS_NAMES

SUGAR_SMARTS: List[str] = [
    "[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]",
    "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]",
    "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]",
    "[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    "[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
]

SUGARS: List[Mol] = [MolFromSmarts(sugar) for sugar in SUGAR_SMARTS]


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
        return any(self.molecule.HasSubstructMatch(sugar) for sugar in SUGARS)

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
        morgan_fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        molecule_with_hydrogens = AddHs(self.molecule)
        morgan_fingerprint = morgan_fingerprint_generator.GetFingerprintAsNumPy(
            mol=molecule_with_hydrogens
        )

        rdkit_fingerprint_generator = rdFingerprintGenerator.GetRDKitFPGenerator()
        rdkit_fingerprint = rdkit_fingerprint_generator.GetFingerprintAsNumPy(
            mol=molecule_with_hydrogens
        )

        mac_keys = MACCSkeys.GenMACCSKeys(molecule_with_hydrogens)
        mac_keys_array: np.ndarray = np.frombuffer(mac_keys.ToBitString().encode(), 'u1') - ord('0')

        # Next, we compute an ensemble of molecular descriptors. Sadly, most RDKIT descriptors are
        # very damn buggy, so we have to be very careful with them and select only the ones that
        # are not buggy. This of course limits the number of descriptors we can use and may change
        # in the future.

        molecular_descriptors = [
            Descriptors.MolWt(molecule_with_hydrogens),
            Descriptors.NumValenceElectrons(molecule_with_hydrogens),
            Descriptors.NumRadicalElectrons(molecule_with_hydrogens),
            rdMolDescriptors.CalcTPSA(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumAromaticRings(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumAliphaticRings(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumSaturatedRings(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumHeteroatoms(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumHeterocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumRotatableBonds(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumSpiroAtoms(molecule_with_hydrogens),
            rdMolDescriptors.CalcFractionCSP3(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumRings(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumAromaticCarbocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumAromaticHeterocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule_with_hydrogens),
            rdMolDescriptors.CalcNumHeavyAtoms(molecule_with_hydrogens),
        ]

        # Crippen descriptors

        molecular_descriptors.extend([
            Crippen.MolLogP(molecule_with_hydrogens),
            Crippen.MolMR(molecule_with_hydrogens),
        ])

        # Lipinski descriptors

        molecular_descriptors.extend([
            Lipinski.NumHDonors(molecule_with_hydrogens),
            Lipinski.NumHAcceptors(molecule_with_hydrogens),
            Lipinski.NumRotatableBonds(molecule_with_hydrogens),
        ])

        molecular_descriptors = np.array(molecular_descriptors, dtype=np.float32)

        # We check that the descriptors are not NaN or infinite
        assert not np.isnan(molecular_descriptors).any(), f"Found NaN: {molecular_descriptors}"
        assert not np.isinf(molecular_descriptors).any(), f"Found INF: {molecular_descriptors}"

        return {
            "morgan_fingerprint": morgan_fingerprint,
            "rdkit_fingerprint": rdkit_fingerprint,
            "mac_keys": mac_keys_array,
            "descriptors": molecular_descriptors
        }

    def labels(self) -> Dict[str, np.ndarray]:
        """Return the labels."""
        return {
            "class": self.one_hot_class,
            "pathway": self.one_hot_pathway,
            "superclass": self.one_hot_superclass,
        }
