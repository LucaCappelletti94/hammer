"""Submodule for fingerprint calculation."""

from typing import Tuple
import numpy as np
from rdkit.Chem import Mol, AddHs  # pylint: disable=no-name-in-module
from rdkit.Chem import rdFingerprintGenerator  # pylint: disable=no-name-in-module


def to_morgan_fingerprint(
    molecule: Mol, radius: int, n_bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a fingerprint for a given molecule defined by a smile.

    Parameters
    ----------
    smiles : str
        The smile of the molecule.
    radius : int
        The radius of the fingerprint.
    n_bits : int
        The number of bits of the fingerprint

    """
    formula = np.zeros((n_bits), np.float32)
    binary = np.zeros((n_bits * radius), np.float32)

    molecule_with_hydrogens = AddHs(molecule)
    for r in range(radius + 1):
        fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=r, fpSize=n_bits
        )
        additional_output = rdFingerprintGenerator.AdditionalOutput()
        additional_output.AllocateBitPaths()
        additional_output.AllocateBitInfoMap()
        fingerprint = fingerprint_generator.GetFingerprint(
            mol=molecule_with_hydrogens, additionalOutput=additional_output
        )
        bit_info_map = additional_output.GetBitInfoMap()
        mol_bi_QC = []
        for i in fingerprint.GetOnBits():
            num_ = len(bit_info_map[i])
            for j in range(num_):
                if bit_info_map[i][j][1] == r:
                    mol_bi_QC.append(i)
                    break

        if r == 0:
            for i in mol_bi_QC:
                formula[i] = len([k for k in bit_info_map[i] if k[1] == 0])
        else:
            for i in mol_bi_QC:
                binary[(n_bits * (r - 1)) + i] = len(
                    [k for k in bit_info_map[i] if k[1] == r]
                )

    return formula.reshape(1, n_bits), binary.reshape(1, n_bits * radius)
