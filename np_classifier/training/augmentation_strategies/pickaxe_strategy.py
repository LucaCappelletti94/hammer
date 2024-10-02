"""Template for a pickaxe run."""

from typing import List
from rdkit.Chem import MolFromSmiles # pylint: disable=no-name-in-module
from minedatabase.pickaxe import Pickaxe
from minedatabase.rules import metacyc_generalized, metacyc_intermediate, BNICE
from np_classifier.training.augmentation_strategies.augmentation_strategy import (
    AugmentationStrategy,
)
from np_classifier.training.molecule import Molecule


class PickaxeStrategy(AugmentationStrategy):
    """Pickaxe strategy for data augmentation"""

    def __init__(
        self,
        rules: str = "BNICE",
        explicit_h: bool = False,
        kekulize: bool = True,
        neutralise: bool = True,
        inchikey_blocks_for_cid: int = 1,
    ):
        """Initializes the Pickaxe strategy.
        
        Parameters
        ----------
        rules: str = "BNICE"
            The rules to use for the pickaxe strategy. Options are "metacyc_generalized", "metacyc_intermediate", "BNICE".
        explicit_h: bool = False
            Whether to add explicit hydrogens.
        kekulize: bool = True
            Whether to kekulize the molecule.
        neutralise: bool = True
            Whether to neutralise the molecule.
        inchikey_blocks_for_cid: int = 1
            The number of blocks to use for the InChIKey.
        """

        if rules == "metacyc_generalized":
            rule_list, coreactant_list, rule_name = metacyc_generalized(
                n_rules=None,
                fraction_coverage=0.2,
                anaerobic=True,
                # exclude_containing = ["aromatic", "halogen"]
            )
        elif rules == "metacyc_intermediate":
            rule_list, coreactant_list, rule_name = metacyc_intermediate(
                n_rules=None,
                fraction_coverage=0.2,
                anaerobic=True,
                # exclude_containing = ["aromatic", "halogen"]
            )
        elif rules == "BNICE":
            rule_list, coreactant_list, rule_name = BNICE()
        else:
            raise ValueError(f"Invalid rules: {rules}")

        self._rule_list = rule_list
        self._coreactant_list = coreactant_list
        self._rule_name = rule_name
        self._explicit_h = explicit_h
        self._kekulize = kekulize
        self._neutralise = neutralise
        self._inchikey_blocks_for_cid = inchikey_blocks_for_cid

    def name(self) -> str:
        return "Pickaxe"

    def augment(self, molecule: Molecule) -> List[Molecule]:
        """Retunrs a list of augmented molecules using the pickaxe strategy."""
        pk = Pickaxe(
            coreactant_list=self._coreactant_list,
            rule_list=self._rule_list,
            explicit_h=self._explicit_h,
            kekulize=self._kekulize,
            neutralise=self._neutralise,
            inchikey_blocks_for_cid=self._inchikey_blocks_for_cid,
            errors=False,
            quiet=True,
            react_targets=False,
        )

        pk.add_molecule(molecule.molecule)
        pk.transform_all()

        augmented_molecules: List[Molecule] = []

        for _id, compound in pk.compounds.items():
            augmented_molecules.append(
                molecule.into_homologue(MolFromSmiles(compound["SMILES"]))
            )

        return augmented_molecules
