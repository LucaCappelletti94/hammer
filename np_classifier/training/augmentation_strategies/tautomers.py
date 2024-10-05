"""Submodule providing an augmentation strategy to generate tautomers."""

from typing import List
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import MolToSmiles, MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import SanitizeMol # pylint: disable=no-name-in-module
from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class TautomersAugmentationStrategy(AugmentationStrategy):
    """Generate tautomers of a molecule."""

    def __init__(self, maximal_number_of_tautomers: int = 64):
        """
        Parameters
        ----------
        maximal_number_of_tautomers: int = 64
            The maximal number of tautomers to generate.
        """
        super().__init__()
        self._maximal_number_of_tautomers = maximal_number_of_tautomers

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Tautomers"

    def augment(self, smiles: str) -> List[str]:
        """Generate tautomers of a molecule."""
        molecule: Mol = MolFromSmiles(smiles)

        enumerator = TautomerEnumerator()
        enumerator.SetMaxTautomers(self._maximal_number_of_tautomers)
        augmented_molecules: List[Mol] = list(enumerator.Enumerate(molecule))

        assert len(augmented_molecules) <= self._maximal_number_of_tautomers

        for augmented_molecule in augmented_molecules:
            SanitizeMol(augmented_molecule)

        augmented_smiles: List[str] = [
            MolToSmiles(homologue, isomericSmiles=True, canonical=True)
            for homologue in augmented_molecules
        ]

        # Ensure the original SMILES is not in the list of augmented SMILES
        if smiles in augmented_smiles:
            augmented_smiles.remove(smiles)

        return augmented_smiles
