"""Submodule implementing the NPCDataset class, offering a corrected version of the NPC dataset."""

from typing import Iterator, Optional, Tuple
import os
import compress_json
import pandas as pd
import numpy as np
from hammer.dags import NPCDAG
from hammer.datasets.dataset import Dataset


class NPCDataset(Dataset):
    """Class defining the NPC dataset."""

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
        return "NPC"

    @staticmethod
    def description() -> str:
        """Return the description of the NPC dataset."""
        return "The NP Classifier dataset with minor corrections."

    def layered_dag(self) -> NPCDAG:
        """Return the Layered DAG for the NPC dataset."""
        return self._layered_dag

    def number_of_samples(self) -> int:
        """Return the number of labeled SMILES in the NPC dataset."""
        return sum(1 for _ in self.iter_samples())

    def iter_samples(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Return an iterator over the labeled SMILES in the NPC dataset."""
        local_path = os.path.join(os.path.dirname(__file__), "categorical.csv.gz")
        categoricals = pd.read_csv(local_path, compression="gzip")
        graph: NPCDAG = self.layered_dag()

        for _, entry in categoricals.iterrows():
            labels = np.zeros(graph.number_of_nodes(), dtype=np.uint8)
            labels[graph.node_id(entry["pathway_label"])] = 1
            labels[graph.node_id(entry["superclass_label"])] = 1
            labels[graph.node_id(entry["class_label"])] = 1
            yield (entry["smiles"], labels)

        multi_labels = compress_json.local_load("multi_label.json")
        relabelled = compress_json.local_load("relabelled.json")

        for entry in multi_labels + relabelled:
            labels = np.zeros(graph.number_of_nodes(), dtype=np.uint8)
            for node_label in entry["pathway_labels"]:
                labels[graph.node_id(node_label)] = 1
            for node_label in entry["superclass_labels"]:
                labels[graph.node_id(node_label)] = 1
            for node_label in entry["class_labels"]:
                labels[graph.node_id(node_label)] = 1

            yield (
                entry["smiles"],
                labels,
            )
