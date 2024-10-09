"""Submodule providing an augmentation strategy based on precomputed Pickaxe molecules."""

from typing import List
import compress_json
from tqdm.auto import tqdm

from hammer.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)


class PickaxeAugmentationStrategy(AugmentationStrategy):
    """Generate molecules from a precomputed Pickaxe file."""

    def __init__(self, pickaxe_path: str, verbose: bool = True):
        """
        Parameters
        ----------
        pickaxe_path: str
            The path to the pickaxe file.
        verbose: bool = True
            Whether to display a progress
        """
        self._pickaxe = compress_json.load(pickaxe_path)
        self._verbose = verbose

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
        return self._pickaxe.get(smiles, [])

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
