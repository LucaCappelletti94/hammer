"""Submodule providing an augmentation strategy to generate stereoisomers."""

from typing import List
from rdkit.Chem import MolToSmiles, MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import SanitizeMol  # pylint: disable=no-name-in-module
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)  # pylint: disable=no-name-in-module
from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class StereoisomersAugmentationStrategy(AugmentationStrategy):
    """Generate stereoisomers of a molecule."""

    def __init__(
        self,
        maximal_number_of_stereoisomers: int = 64,
        only_stereo_groups: bool = True,
    ):
        """

        Parameters
        ----------
        maximal_number_of_stereoisomers: int = 64
            The maximal number of stereoisomers to generate.
        only_stereo_groups: bool = False
            Only find stereoisomers that differ at the StereoGroups associated with the molecule.
        """
        self._stereo_options = StereoEnumerationOptions(
            tryEmbedding=False,
            onlyStereoGroups=only_stereo_groups,
            unique=True,
            maxIsomers=maximal_number_of_stereoisomers,
        )
        self._maximal_number_of_stereoisomers = maximal_number_of_stereoisomers

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Stereoisomers"

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

            assert len(augmented_smiles) <= self._maximal_number_of_stereoisomers

            # Ensure the original SMILES is not in the list of augmented SMILES
            if smiles in augmented_smiles:
                augmented_smiles.remove(smiles)

            return augmented_smiles
        except RuntimeError as _runtime_error:
            # raise RuntimeError(
            #     f"Error generating stereoisomers for {molecule.smiles}"
            # ) from runtime_error
            return []
