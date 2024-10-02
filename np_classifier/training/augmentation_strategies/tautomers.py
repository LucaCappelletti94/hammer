"""Submodule providing an augmentation strategy to generate tautomers."""

from typing import List
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)
from np_classifier.training.molecule import Molecule


class TautomersAugmentationStrategy(AugmentationStrategy):
    """Generate tautomers of a molecule."""

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Tautomers"

    def augment(self, molecule: Molecule) -> List[Molecule]:
        """Generate tautomers of a molecule."""
        return [
            molecule.into_homologue(homologue)
            for homologue in TautomerEnumerator().Enumerate(molecule.molecule)
        ]
