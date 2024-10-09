"""Submodule providing an augmentation strategy to generate tautomers."""

from typing import List, Optional
from multiprocessing import Pool
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import MolToSmiles, MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import SanitizeMol  # pylint: disable=no-name-in-module
from rdkit import RDLogger
from tqdm.auto import tqdm
from hammer.training.augmentation_strategies.augmentation_strategy import (
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
        super().__init__()
        self._maximal_number = maximal_number
        self._verbose = verbose
        self._n_jobs = n_jobs

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

    def augment_all(self, smiles: List[str]) -> List[List[str]]:
        """Generate tautomers of a list of molecules."""
        RDLogger.DisableLog("rdApp.*")
        with Pool(self._n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(self.augment, smiles),
                    total=len(smiles),
                    disable=not self._verbose,
                    desc=self.name(),
                    dynamic_ncols=True,
                    leave=False,
                )
            )
            pool.close()
            pool.join()
        
        RDLogger.EnableLog("rdApp.*")

        return results
