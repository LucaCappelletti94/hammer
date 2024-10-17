"""Submodule providing a variant of the NPC Dataset labelled by running NP Classifier on its own SMILES."""

from typing import Iterator, Optional
import compress_json
from hammer.layered_dags import NPCLayeredDAG
from hammer.datasets.smiles_dataset import Dataset
from hammer.datasets.labeled_smiles import LabeledSMILES


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
        self._layered_dag = NPCLayeredDAG()

    @staticmethod
    def name() -> str:
        """Return the name of the NPC dataset."""
        return "NPCScraped"

    @staticmethod
    def description() -> str:
        """Return the description of the NPC dataset."""
        return "The NPC dataset labelled by running NP Classifier on its own SMILES."

    @staticmethod
    def multi_label() -> bool:
        """Return whether the NPC dataset is multi-label."""
        return True

    def layered_dag(self) -> NPCLayeredDAG:
        """Return the Layered DAG for the NPC dataset."""
        return self._layered_dag

    def number_of_smiles(self) -> int:
        """Return the number of labeled SMILES in the NPC dataset."""
        return sum(1 for _ in self.iter_labeled_smiles())

    def iter_labeled_smiles(self) -> Iterator[LabeledSMILES]:
        """Return an iterator over the labeled SMILES in the NPC dataset."""
        for entry in compress_json.local_load("npc-scraped.json.gz"):
            yield LabeledSMILES(
                entry["smiles"],
                {
                    "pathways": entry["pathway_results"],
                    "superclasses": entry["superclass_results"],
                    "classes": entry["class_results"],
                },
            )
