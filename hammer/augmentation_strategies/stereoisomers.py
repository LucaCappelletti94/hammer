"""Submodule providing an augmentation strategy to generate stereoisomers."""

from typing import List
from rdkit.Chem import MolToSmiles, MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import SanitizeMol  # pylint: disable=no-name-in-module
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from hammer.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class StereoisomersAugmentationStrategy(AugmentationStrategy):
    """Generate stereoisomers of a molecule."""

    def __init__(
        self,
        maximal_number: int = 64,
        n_jobs: int = None,
        verbose: bool = True,
    ):
        """

        Parameters
        ----------
        maximal_number: int = 64
            The maximal number of stereoisomers to generate.
        n_jobs: Optional[int] = None
            The number of jobs to use for parallel processing.
        verbose: bool = True
            Whether to display a progress bar.
        """
        self._stereo_options = StereoEnumerationOptions(
            tryEmbedding=False,
            onlyStereoGroups=True,
            unique=True,
            maxIsomers=maximal_number,
        )
        super().__init__(n_jobs=n_jobs, verbose=verbose)

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Stereoisomers"

    @staticmethod
    def pythonic_name() -> str:
        """Return the pythonic name of the augmentation strategy."""
        return "stereoisomers"

    @staticmethod
    def argparse_description() -> str:
        """Return the argparse description of the augmentation strategy."""
        return "Augments the dataset with stereoisomers of the original SMILES."

    def augment(self, smiles: str) -> List[str]:
        """Generate stereoisomers of a molecule."""
        try:
            molecule: Mol = MolFromSmiles(smiles)
            augmented_molecules: List[Mol] = list(
                EnumerateStereoisomers(molecule, options=self._stereo_options)
            )

            for augmented_molecule in augmented_molecules:
                SanitizeMol(augmented_molecule)

            augmented_smiles: List[str] = [
                MolToSmiles(homologue, isomericSmiles=True, canonical=True)
                for homologue in EnumerateStereoisomers(
                    MolFromSmiles(smiles), options=self._stereo_options
                )
            ]

            # Ensure the original SMILES is not in the list of augmented SMILES
            if smiles in augmented_smiles:
                augmented_smiles.remove(smiles)

            return augmented_smiles
        except RuntimeError as _runtime_error:
            # raise RuntimeError(
            #     f"Error generating stereoisomers for {molecule.smiles}"
            # ) from runtime_error
            return []
