"""Submodule providing an augmentation strategy to generate tautomers."""

from typing import List
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import MolToSmiles, MolFromSmiles  # pylint: disable=no-name-in-module
from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class TautomersAugmentationStrategy(AugmentationStrategy):
    """Generate tautomers of a molecule."""

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Tautomers"

    def augment(self, smiles: str) -> List[str]:
        """Generate tautomers of a molecule."""
        augmented_smiles: List[str] = [
            MolToSmiles(homologue, isomericSmiles=True)
            for homologue in TautomerEnumerator().Enumerate(MolFromSmiles(smiles))
        ]

        # Ensure the original SMILES is not in the list of augmented SMILES
        if smiles in augmented_smiles:
            augmented_smiles.remove(smiles)

        return augmented_smiles
