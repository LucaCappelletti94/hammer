"""Submodule providing an augmentation strategy to generate tautomers."""

from typing import List, Optional
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import MolToSmiles, MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import SanitizeMol  # pylint: disable=no-name-in-module
from hammer.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class TautomersAugmentationStrategy(AugmentationStrategy):
    """Generate tautomers of a molecule."""

    def __init__(
        self,
        maximal_number: int = 64,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        maximal_number: int = 64
            The maximal number of tautomers to generate.
        n_jobs: Optional[int] = None
            The number of jobs to use for parallel processing.
        verbose: bool = True
            Whether to display a progress bar.
        """
        self._maximal_number = maximal_number
        super().__init__(n_jobs=n_jobs, verbose=verbose)

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Tautomers"

    @staticmethod
    def pythonic_name() -> str:
        """Return the pythonic name of the augmentation strategy."""
        return "tautomers"

    @staticmethod
    def argparse_description() -> str:
        """Return the argparse description of the augmentation strategy."""
        return "Augments the dataset with tautomers of the original SMILES."

    def augment(self, smiles: str) -> List[str]:
        """Generate tautomers of a molecule."""
        molecule: Mol = MolFromSmiles(smiles)

        enumerator = TautomerEnumerator()
        enumerator.SetMaxTautomers(self._maximal_number)
        augmented_molecules: List[Mol] = list(enumerator.Enumerate(molecule))

        assert len(augmented_molecules) <= self._maximal_number

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
