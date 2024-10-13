"""Submodule implementing the NPCDataset class, offering a corrected version of the NPC dataset."""

from typing import Iterator, Optional
import os
import compress_json
import pandas as pd
from hammer.training.layered_dags import NPCLayeredDAG
from hammer.training.datasets.smiles_dataset import Dataset
from hammer.training.datasets.labeled_smiles import LabeledSMILES


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
        self._layered_dag = NPCLayeredDAG()

    @staticmethod
    def name() -> str:
        """Return the name of the NPC dataset."""
        return "NPC"

    def layered_dag(self) -> NPCLayeredDAG:
        """Return the Layered DAG for the NPC dataset."""
        return self._layered_dag

    def number_of_smiles(self) -> int:
        """Return the number of labeled SMILES in the NPC dataset."""
        return sum(1 for _ in self.iter_labeled_smiles())

    def iter_labeled_smiles(self) -> Iterator[LabeledSMILES]:
        """Return an iterator over the labeled SMILES in the NPC dataset."""
        local_path = os.path.join(os.path.dirname(__file__), "categorical.csv.gz")
        categoricals = pd.read_csv(local_path, compression="gzip")

        for entry in categoricals.itertuples():
            yield LabeledSMILES(
                entry.smiles,
                {
                    "pathways": [entry.pathway_label],
                    "superclasses": [entry.superclass_label],
                    "classes": [entry.class_label],
                },
            )

        multi_labels = compress_json.local_load("multi_label.json")
        relabelled = compress_json.local_load("relabelled.json")

        for entry in multi_labels + relabelled:
            yield LabeledSMILES(
                entry["smiles"],
                {
                    "pathways": entry["pathway_labels"],
                    "superclasses": entry["superclass_labels"],
                    "classes": entry["class_labels"],
                },
            )
