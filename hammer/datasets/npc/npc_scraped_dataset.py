"""Submodule providing a variant of the NPC Dataset labelled by running NP Classifier on its own SMILES."""

from typing import Iterator, Optional, Tuple
import compress_json
import numpy as np
from hammer.datasets.dataset import Dataset
from hammer.dags import NPCDAG


class NPCScrapedDataset(Dataset):
    """Class defining the NPC dataset labelled by running NP Classifier on its own SMILES."""

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
        return "NPCScraped"

    @staticmethod
    def description() -> str:
        """Return the description of the NPC dataset."""
        return "The NPC dataset labelled by running NP Classifier on its own SMILES."

    def layered_dag(self) -> NPCDAG:
        """Return the Layered DAG for the NPC dataset."""
        return self._layered_dag

    def number_of_samples(self) -> int:
        """Return the number of labeled SMILES in the NPC dataset."""
        return sum(1 for _ in self.iter_samples())

    def iter_samples(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Return an iterator over the labeled SMILES in the NPC dataset."""
        for entry in compress_json.local_load("npc-scraped.json.gz"):
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
