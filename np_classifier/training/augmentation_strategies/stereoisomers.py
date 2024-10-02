"""Submodule providing an augmentation strategy to generate stereoisomers."""

from typing import List
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)  # pylint: disable=no-name-in-module
from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)
from np_classifier.training.molecule import Molecule


class StereoisomersAugmentationStrategy(AugmentationStrategy):
    """Generate stereoisomers of a molecule."""

    def __init__(
        self,
        only_stereo_groups: bool = True,
    ):
        """

        Parameters
        ----------
        only_stereo_groups: bool = False
            Only find stereoisomers that differ at the StereoGroups associated with the molecule.
        """
        self._stereo_options = StereoEnumerationOptions(
            tryEmbedding=False,
            onlyStereoGroups=only_stereo_groups,
            unique=True,
            maxIsomers=64
        )

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Stereoisomers"

    def augment(self, molecule: Molecule) -> List[Molecule]:
        """Generate stereoisomers of a molecule."""
        try:
            return [
                molecule.into_homologue(homologue)
                for homologue in EnumerateStereoisomers(
                    molecule.molecule, options=self._stereo_options
                )
            ]
        except RuntimeError as _runtime_error:
            # raise RuntimeError(
            #     f"Error generating stereoisomers for {molecule.smiles}"
            # ) from runtime_error
            return []
