"""Submodule for SMILES dataclass."""

from typing import List, Dict
import numpy as np
from rdkit.Chem import Mol  # pylint: disable=no-name-in-module
from rdkit.Chem import MolFromSmiles  # pylint: disable=no-name-in-module
from rdkit.Chem import MolFromSmarts  # pylint: disable=no-name-in-module
from rdkit.Chem.rdFingerprintGenerator import (  # pylint: disable=no-name-in-module
    GetMorganGenerator,  # pylint: disable=no-name-in-module
    GetRDKitFPGenerator,  # pylint: disable=no-name-in-module
    GetAtomPairGenerator,  # pylint: disable=no-name-in-module
    GetTopologicalTorsionGenerator,  # pylint: disable=no-name-in-module
    GetMorganFeatureAtomInvGen,  # pylint: disable=no-name-in-module
)
from rdkit.Chem import AddHs  # pylint: disable=no-name-in-module
from rdkit.Chem import MACCSkeys  # pylint: disable=no-name-in-module
from rdkit.Avalon.pyAvalonTools import GetAvalonFP  # pylint: disable=no-name-in-module
from rdkit.Chem import Descriptors  # pylint: disable=no-name-in-module
from rdkit.Chem import Lipinski  # pylint: disable=no-name-in-module
from rdkit.Chem import Crippen  # pylint: disable=no-name-in-module
from rdkit.Chem.rdMolDescriptors import (  # pylint: disable=no-name-in-module
    CalcTPSA,  # pylint: disable=no-name-in-module
    CalcNumAromaticRings,  # pylint: disable=no-name-in-module
    CalcNumAliphaticRings,  # pylint: disable=no-name-in-module
    CalcNumSaturatedRings,  # pylint: disable=no-name-in-module
    CalcNumHeteroatoms,  # pylint: disable=no-name-in-module
    CalcNumHeterocycles,  # pylint: disable=no-name-in-module
    CalcNumRotatableBonds,  # pylint: disable=no-name-in-module
    CalcNumSpiroAtoms,  # pylint: disable=no-name-in-module
    CalcFractionCSP3,  # pylint: disable=no-name-in-module
    CalcNumRings,  # pylint: disable=no-name-in-module
    CalcNumAromaticCarbocycles,  # pylint: disable=no-name-in-module
    CalcNumAromaticHeterocycles,  # pylint: disable=no-name-in-module
    CalcNumAliphaticCarbocycles,  # pylint: disable=no-name-in-module
    CalcNumAliphaticHeterocycles,  # pylint: disable=no-name-in-module
    CalcNumSaturatedCarbocycles,  # pylint: disable=no-name-in-module
    CalcNumSaturatedHeterocycles,  # pylint: disable=no-name-in-module
    CalcNumHeavyAtoms,  # pylint: disable=no-name-in-module
)
from rdkit.Chem import GraphDescriptors  # pylint: disable=no-name-in-module
from map4 import MAP4

SUGAR_SMARTS: List[str] = [
    "[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]",
    "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]",
    "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]",
    "[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    "[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
]

SUGARS: List[Mol] = [MolFromSmarts(sugar) for sugar in SUGAR_SMARTS]


def compute_features(
    smile: str,
    radius: int = 3,
    n_bits: int = 2048,
    include_morgan_fingerprint: bool = True,
    include_rdkit_fingerprint: bool = True,
    include_atom_pair_fingerprint: bool = True,
    include_topological_torsion_fingerprint: bool = True,
    include_feature_morgan_fingerprint: bool = True,
    include_avalon_fingerprint: bool = True,
    include_maccs_fingerprint: bool = True,
    include_map4_fingerprint: bool = True,
    include_descriptors: bool = True,
) -> Dict[str, np.ndarray]:
    """Return a complete set of fingerprints and descriptors.

    Parameters
    ----------
    smile : str
        The SMILES of the molecule.
    radius : int
        The radius of the Morgan fingerprint.
    n_bits : int
        The number of bits of the fingerprints.
    include_morgan_fingerprint : bool
        Whether to include the Morgan fingerprint.
    include_rdkit_fingerprint : bool
        Whether to include the RDKit fingerprint.
    include_atom_pair_fingerprint : bool
        Whether to include the Atom Pair fingerprint.
    include_topological_torsion_fingerprint : bool
        Whether to include the Topological Torsion fingerprint.
    include_feature_morgan_fingerprint : bool
        Whether to include the Feature Morgan fingerprint.
    include_avalon_fingerprint : bool
        Whether to include the Avalon fingerprint.
    include_maccs_fingerprint : bool
        Whether to include the MACCS fingerprint.
    include_map4_fingerprint : bool
        Whether to include the MAP4 fingerprint.
    include_descriptors : bool
        Whether to include the molecular descriptors.
    """
    features = {}
    molecule = MolFromSmiles(smile)
    molecule_with_hydrogens = AddHs(molecule)

    if include_morgan_fingerprint:
        morgan_fingerprint_generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        morgan_fingerprint = morgan_fingerprint_generator.GetFingerprintAsNumPy(
            mol=molecule_with_hydrogens
        )

        features["morgan_fingerprint"] = morgan_fingerprint.astype(np.uint8)

    if include_rdkit_fingerprint:
        rdkit_fingerprint_generator = GetRDKitFPGenerator(fpSize=n_bits)
        rdkit_fingerprint = rdkit_fingerprint_generator.GetFingerprintAsNumPy(
            mol=molecule_with_hydrogens
        )

        features["rdkit_fingerprint"] = rdkit_fingerprint.astype(np.uint8)

    if include_atom_pair_fingerprint:
        atom_pair_fingerprint_generator = GetAtomPairGenerator(fpSize=n_bits)
        atom_pair_fingerprint = atom_pair_fingerprint_generator.GetFingerprintAsNumPy(
            mol=molecule_with_hydrogens
        )

        features["atom_pair_fingerprint"] = atom_pair_fingerprint.astype(np.uint8)

    if include_topological_torsion_fingerprint:
        topological_torsion_fingerprint_generator = GetTopologicalTorsionGenerator(
            fpSize=n_bits
        )
        topological_torsion_fingerprint = (
            topological_torsion_fingerprint_generator.GetFingerprintAsNumPy(
                mol=molecule_with_hydrogens
            )
        )

        features["topological_torsion_fingerprint"] = topological_torsion_fingerprint.astype(np.uint8)

    if include_feature_morgan_fingerprint:
        feature_morgan_fingerprint_generator = GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            atomInvariantsGenerator=GetMorganFeatureAtomInvGen(),
        )
        feature_morgan_fingerprint = (
            feature_morgan_fingerprint_generator.GetFingerprintAsNumPy(
                mol=molecule_with_hydrogens
            )
        )

        features["feature_morgan_fingerprint"] = feature_morgan_fingerprint.astype(np.uint8)

    if include_avalon_fingerprint:
        # We proceed generating the Avalon fingerprints
        avalon_fingerprint: "ExplicitBitVect" = GetAvalonFP(
            molecule_with_hydrogens, nBits=n_bits
        )
        avalon_fingerprint_array = np.frombuffer(
            avalon_fingerprint.ToBitString().encode(), "u1"
        ) - ord("0")

        features["avalon_fingerprint"] = avalon_fingerprint_array.astype(np.uint8)

    if include_map4_fingerprint:
        map4 = MAP4(dimensions=n_bits, radius=radius)
        map4_fingerprint = map4.calculate(molecule_with_hydrogens)

        features["map4_fingerprint"] = map4_fingerprint.astype(np.uint8)

    if include_maccs_fingerprint:
        maccs_fingerprint: "ExplicitBitVect" = MACCSkeys.GenMACCSKeys(
            molecule_with_hydrogens
        )
        maccs_fingerprint_array: np.ndarray = np.frombuffer(
            maccs_fingerprint.ToBitString().encode(), "u1"
        ) - ord("0")

        features["maccs_fingerprint"] = maccs_fingerprint_array.astype(np.uint8)

    if include_descriptors:
        # Next, we compute an ensemble of molecular descriptors. Sadly, most RDKIT descriptors are
        # very damn buggy, so we have to be very careful with them and select only the ones that
        # are not buggy. This of course limits the number of descriptors we can use and may change
        # in the future.

        molecular_descriptors = [
            Descriptors.MolWt(molecule_with_hydrogens),
            Descriptors.NumValenceElectrons(molecule_with_hydrogens),
            Descriptors.NumRadicalElectrons(molecule_with_hydrogens),
            CalcTPSA(molecule_with_hydrogens),
            CalcNumAromaticRings(molecule_with_hydrogens),
            CalcNumAliphaticRings(molecule_with_hydrogens),
            CalcNumSaturatedRings(molecule_with_hydrogens),
            CalcNumHeteroatoms(molecule_with_hydrogens),
            CalcNumHeterocycles(molecule_with_hydrogens),
            CalcNumRotatableBonds(molecule_with_hydrogens),
            CalcNumSpiroAtoms(molecule_with_hydrogens),
            CalcFractionCSP3(molecule_with_hydrogens),
            CalcNumRings(molecule_with_hydrogens),
            CalcNumAromaticCarbocycles(molecule_with_hydrogens),
            CalcNumAromaticHeterocycles(molecule_with_hydrogens),
            CalcNumAliphaticCarbocycles(molecule_with_hydrogens),
            CalcNumAliphaticHeterocycles(molecule_with_hydrogens),
            CalcNumSaturatedCarbocycles(molecule_with_hydrogens),
            CalcNumSaturatedHeterocycles(molecule_with_hydrogens),
            CalcNumHeavyAtoms(molecule_with_hydrogens),
            any(molecule.HasSubstructMatch(sugar) for sugar in SUGARS),
        ]

        # Graph descriptors
        molecular_descriptors.extend(
            [
                GraphDescriptors.BalabanJ(molecule_with_hydrogens),
                GraphDescriptors.BertzCT(molecule_with_hydrogens),
            ]
        )

        # Crippen descriptors

        molecular_descriptors.extend(
            [
                Crippen.MolLogP(molecule_with_hydrogens),
                Crippen.MolMR(molecule_with_hydrogens),
            ]
        )

        # Lipinski descriptors

        molecular_descriptors.extend(
            [
                Lipinski.NumHDonors(molecule_with_hydrogens),
                Lipinski.NumHAcceptors(molecule_with_hydrogens),
                Lipinski.NumRotatableBonds(molecule_with_hydrogens),
            ]
        )

        molecular_descriptors = np.array(molecular_descriptors, dtype=np.float32)
        assert molecular_descriptors.size == len(
            descriptor_names()
        ), f"Expected {len(descriptor_names())}, got {molecular_descriptors.size}"

        # We check that the descriptors are not NaN or infinite
        assert not np.isnan(
            molecular_descriptors
        ).any(), f"Found NaN: {molecular_descriptors}"
        assert not np.isinf(
            molecular_descriptors
        ).any(), f"Found INF: {molecular_descriptors}"

        features["descriptors"] = molecular_descriptors

    return features


def descriptor_names() -> List[str]:
    """Return the names of the descriptors."""
    return [
        "MolWt",
        "NumValenceElectrons",
        "NumRadicalElectrons",
        "TPSA",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumHeteroatoms",
        "NumHeterocycles",
        "NumRotatableBonds",
        "NumSpiroAtoms",
        "FractionCSP3",
        "NumRings",
        "NumAromaticCarbocycles",
        "NumAromaticHeterocycles",
        "NumAliphaticCarbocycles",
        "NumAliphaticHeterocycles",
        "NumSaturatedCarbocycles",
        "NumSaturatedHeterocycles",
        "NumHeavyAtoms",
        "ContainsSugar",
        "BalabanJ",
        "BertzCT",
        "MolLogP",
        "MolMR",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
    ]
