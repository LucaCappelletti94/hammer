"""Submodule providing an harmonized variant of the NPC Dataset."""

from typing import Iterator, Optional, Tuple, List
import compress_json
import numpy as np
from hammer.dags import NPCDAG
from hammer.datasets.dataset import Dataset

RARE_TERMS: List[str] = [
    "Cyclogermacrane sesquiterpenoids",
    "Carotenoids (C40, Χ-Χ)",
    "Carotenoids (C45, β-Ψ)",
    "Carotenoids (C50, β-β)",
    "Eicosa-1,2-dioxolanes",
    "RiPPs-Bottromycins"
]

class NPCHarmonizedDataset(Dataset):
    """Class defining the harmonized NPC dataset."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        maximal_number_of_molecules: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the NPC dataset."""
        super().__init__(
            random_state=random_state,
            maximal_number_of_molecules=maximal_number_of_molecules,
            verbose=verbose,
        )
        self._layered_dag = NPCDAG()

    @staticmethod
    def name() -> str:
        """Return the name of the NPC dataset."""
        return "NPCHarmonized"

    @staticmethod
    def description() -> str:
        """Return the description of the NPC dataset."""
        return "The harmonized NPC dataset."

    def layered_dag(self) -> NPCDAG:
        """Return the Layered DAG for the NPC dataset."""
        return self._layered_dag

    def number_of_samples(self) -> int:
        """Return the number of labeled SMILES in the NPC dataset."""
        return sum(1 for _ in self.iter_samples())

    def iter_samples(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Return an iterator over the labeled SMILES in the NPC dataset."""
        for entry in compress_json.local_load("npc-harmonized.json.xz"):
            skip_sample: bool = False
            for node_label in entry["pathways"] + entry["superclasses"] + entry["classes"]:
                if node_label in RARE_TERMS:
                    skip_sample = True
                    break
            if skip_sample:
                continue

            labels = np.zeros(self.layered_dag().number_of_nodes(), dtype=np.uint8)

            for node_label in entry["pathways"]:
                labels[self.layered_dag().node_id(node_label)] = 1

            for node_label in entry["superclasses"]:
                labels[self.layered_dag().node_id(node_label)] = 1

            for node_label in entry["classes"]:
                labels[self.layered_dag().node_id(node_label)] = 1

            yield (
                entry["smiles"],
                labels
            )
