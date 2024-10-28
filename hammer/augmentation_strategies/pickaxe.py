"""Submodule providing an augmentation strategy based on precomputed Pickaxe molecules."""

from typing import List, Optional
import random
import compress_json
from tqdm.auto import tqdm
from hammer.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class PickaxeAugmentationStrategy(AugmentationStrategy):
    """Generate molecules from a precomputed Pickaxe file."""

    def __init__(
        self,
        maximal_number: int = 64,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        maximal_number: int = 64
            The maximal number of molecules to generate.
        n_jobs: int = None
            The number of jobs to use for parallel processing.
        verbose: bool = True
            Whether to display a progress
        """
        self._pickaxe = compress_json.local_load("pickaxe_normalized.json.xz")
        super().__init__(maximal_number=maximal_number, n_jobs=n_jobs, verbose=verbose)

    def name(self) -> str:
        """Return the name of the augmentation strategy."""
        return "Pickaxe"

    @staticmethod
    def pythonic_name() -> str:
        """Return the pythonic name of the augmentation strategy."""
        return "pickaxe"

    @staticmethod
    def argparse_description() -> str:
        """Return the argparse description of the augmentation strategy."""
        return "Augments the dataset with molecules from a precomputed Pickaxe file."

    def augment(self, smiles: str) -> List[str]:
        """Augment a smiles."""
        smiles_transformations = self._pickaxe.get(smiles, [])
        if len(smiles_transformations) > self._maximal_number:
            random.shuffle(smiles_transformations)
            smiles_transformations = smiles_transformations[: self._maximal_number]
        return smiles_transformations

    def augment_all(self, smiles: List[str]) -> List[List[str]]:
        """Augment a list of smiles."""
        return [
            self.augment(smiles)
            for smiles in tqdm(
                smiles,
                desc=self.name(),
                disable=not self._verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ]
