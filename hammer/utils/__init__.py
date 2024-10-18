"""Utilities."""

from typing import Sequence, List, Optional
from multiprocessing import Pool
from tqdm.auto import tqdm
from rdkit.Chem import MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem import MolToSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem import SanitizeMol  # pylint: disable=no-name-in-module
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES is valid."""
    RDLogger.DisableLog("rdApp.error")
    valid = MolFromSmiles(smiles) is not None
    RDLogger.EnableLog("rdApp.error")
    return valid


def _smiles_to_molecule(smiles: str) -> Mol:
    """Convert a SMILES to an RDKit molecule."""
    molecule: Mol = MolFromSmiles(smiles)
    SanitizeMol(molecule)
    return molecule


def _into_canonical(smiles: str) -> str:
    """Convert a SMILES to a canonical SMILES."""
    molecule: Mol = MolFromSmiles(smiles)
    SanitizeMol(molecule)
    return MolToSmiles(molecule, isomericSmiles=True, canonical=True)


def smiles_to_molecules(
    smiles: Sequence[str], verbose: bool = True, n_jobs: Optional[int] = None
) -> List[Mol]:
    """Convert a list of SMILES to a list of RDKit molecules."""
    with Pool(n_jobs) as pool:
        molecules = list(
            tqdm(
                pool.imap(_smiles_to_molecule, smiles),
                total=len(smiles),
                disable=not verbose,
                desc="Converting SMILES to molecules",
                dynamic_ncols=True,
                leave=False,
            )
        )
        pool.close()
        pool.join()

    return molecules


def into_canonical_smiles(
    smiles: Sequence[str], verbose: bool = True, n_jobs: Optional[int] = None
) -> List[str]:
    """Convert a list of SMILES to a list of canonical SMILES."""
    with Pool(n_jobs) as pool:
        canonical_smiles = list(
            tqdm(
                pool.imap(_into_canonical, smiles),
                total=len(smiles),
                disable=not verbose,
                desc="Converting SMILES to canonical SMILES",
                dynamic_ncols=True,
                leave=False,
            )
        )
        pool.close()
        pool.join()

    return canonical_smiles


__all__ = [
    "smiles_to_molecules",
    "into_canonical_smiles",
]
