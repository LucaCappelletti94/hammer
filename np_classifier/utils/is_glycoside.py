"""Submodule to determine whether a molecule is a glycoside or not."""

from rdkit.Chem import MolFromSmiles, MolFromSmarts  # pylint: disable=no-name-in-module


def is_glycoside(
    smiles: str,
) -> bool:
    """Returns whether a molecule is a glycoside or not."""
    sugar1 = MolFromSmarts("[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]")
    sugar2 = MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    )
    sugar3 = MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]"
    )
    sugar4 = MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]"
    )
    sugar5 = MolFromSmarts(
        "[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    )
    sugar6 = MolFromSmarts(
        "[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    )
    mol = MolFromSmiles(smiles, sanitize=True)
    return (
        mol.HasSubstructMatch(sugar1)
        or mol.HasSubstructMatch(sugar2)
        or mol.HasSubstructMatch(sugar3)
        or mol.HasSubstructMatch(sugar4)
        or mol.HasSubstructMatch(sugar5)
        or mol.HasSubstructMatch(sugar6)
    )
