"""Submodule providing strategies to extend the smiles in the training dataset."""

import copy

from typing import List
from rdkit.Chem import Mol, Atom  # pylint: disable=no-name-in-module
from rdkit.Chem import AddHs  # pylint: disable=no-name-in-module
from rdkit.Chem import RemoveHs  # pylint: disable=no-name-in-module
from rdkit.Chem import SanitizeMol  # pylint: disable=no-name-in-module


def get_hydroxyl_groups_candidates(molecule: Mol) -> List[Atom]:
    """Returns a list of hydroxyl groups candidates in the molecule.

    Parameters
    ----------
    molecule : Mol
        The molecule to search for hydroxyl groups.
    """
    return [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "H"
        and any(8 == neighbor.GetAtomicNum() for neighbor in atom.GetNeighbors())
    ]


def get_methoxyl_groups_candidates(molecule: Mol) -> List[Atom]:
    """Returns a list of methoxyl groups in the molecule.

    Parameters
    ----------
    mol : Mol
        The molecule to search for methoxyl groups.
    """
    return [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "C"
        and [neighbor.GetAtomicNum() for neighbor in atom.GetNeighbors()] == [8]
    ]


def generate_methoxylated_homologues(molecule: Mol) -> List[Mol]:
    """Returns a list of methoxylated homologues of a molecule.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.
    """
    molecule = AddHs(molecule)
    methoxylated_homologues = []
    number_of_hydroxyl_groups_candidates = len(get_hydroxyl_groups_candidates(molecule))
    for atom_number in range(number_of_hydroxyl_groups_candidates):

        # We need to copy the molecule to avoid modifying the original molecule
        molecule_clone = copy.deepcopy(molecule)
        # We get the hydroxyl groups candidates
        hydroxyl_groups_candidates = get_hydroxyl_groups_candidates(molecule_clone)
        # We change INPLACE the atomic number of the hydroxyl group to carbon
        atom = hydroxyl_groups_candidates[atom_number]
        atom.SetAtomicNum(6)
        # We remove the hydrogens which were added by AddHs
        molecule_clone = RemoveHs(molecule_clone)
        SanitizeMol(molecule_clone)

        methoxylated_homologues.append(molecule_clone)
    return methoxylated_homologues


def generate_demethoxylated_homologues(molecule: Mol) -> List[Mol]:
    """Returns a list of demethoxylated homologues of a molecule.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.
    """
    demethoxylated_homologues = []
    number_of_metoxyl_groups_candidates = len(get_methoxyl_groups_candidates(molecule))
    for atom_number in range(number_of_metoxyl_groups_candidates):

        molecule_clone = copy.deepcopy(molecule)
        metoxyl_groups_candidates = get_methoxyl_groups_candidates(molecule_clone)
        atom = metoxyl_groups_candidates[atom_number]
        atom.SetAtomicNum(8)
        molecule_clone = RemoveHs(molecule_clone)
        SanitizeMol(molecule_clone)

        demethoxylated_homologues.append(molecule_clone)
    return demethoxylated_homologues
