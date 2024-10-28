"""Script to generate an harmonized version of the NPC dataset, which takes into
account the original dataset, the predictions of the scraped and the manual corrections
made by us in the 'divergent NPC entries'. The following script follows these rules:

1) If the original dataset entry is compatible with the scraped prediction, we consider
the scraped prediction as correct, and as the scraped prediction may include more information
due to the fact that it often contains more labels, we use the scraped prediction as the
ground truth for the entry.
2) If the original dataset entry is not compatible with the scraped prediction, we look at the
rules defined in the 'harmonization_rules' function. The function will either return the correct
prediction if the rules can determine one, or will return None if the rules can't determine a
correct prediction.
3) If the original dataset entry is not compatible with the scraped prediction and the rules
can't determine a correct prediction, we look into the 'divergent NPC entries' and see whether
there is a manual correction for the entry. If there is, we use the manual correction as the
ground truth for the entry.
4) If none of the above conditions are met, we discard the entry as we can't determine the
correct prediction for it.
5) If a SMILES only appears in the scraped dataset, and its predictions are complete (there is
no empty set of clases, superclasses or subclasses), we consider the scraped prediction as the
"""

from typing import Dict, Set
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import compress_json
from hammer.datasets import NPCDataset, NPCScrapedDataset

# ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"


def harmonize_npc() -> None:
    """Tries to harmonize the NPC dataset."""

    original: Dict[str, np.ndarray] = {
        smiles: labels for smiles, labels in NPCDataset().iter_samples()
    }
    scraped: Dict[str, np.ndarray] = {
        smiles: labels for smiles, labels in NPCScrapedDataset().iter_samples()
    }

    harmonized: Dict[str, np.ndarray] = {}
    discarded_smiles: Set[str] = set()
    harmonization_tecniques_counts: Dict[str, int] = {
        "scraped_is_empty": 0,
        "no_divergence": 0,
    }

    for smiles, original_entry in original.items():
        scraped_entry = scraped.get(smiles)
        if scraped_entry is None or np.sum(scraped_entry) < 3:
            discarded_smiles.add(smiles)
            harmonization_tecniques_counts["scraped_is_empty"] += 1
            continue

        if np.all(scraped_entry == original_entry):
            harmonized[smiles] = scraped_entry
            harmonization_tecniques_counts["no_divergence"] += 1
            continue

        discarded_smiles.add(smiles)

    print(f"{BOLD}{RED}Discarded:{RESET} {len(discarded_smiles)} entries")

    print(f"{BOLD}{GREEN}Harmonized:{RESET} {len(harmonized)} entries")

    for technique, count in harmonization_tecniques_counts.items():
        print(f"{BLUE}└── {technique}:{RESET} {count}")

    dag = NPCDataset().layered_dag()

    harmonized_df = pd.DataFrame(
        [
            {"smiles": smiles, **dict(zip(NPCDataset().layered_dag().nodes(), labels))}
            for smiles, labels in tqdm(
                harmonized.items(),
                total=len(harmonized),
                desc="Generating harmonized DataFrame",
            )
        ]
    )

    harmonized_df.set_index("smiles", inplace=True)

    compress_json.dump(
        [
            {
                "smiles": smiles,
                "pathway_labels": [
                    pathway
                    for pathway in dag.nodes_in_layer("pathways")
                    if smiles_labels[pathway] == 1
                ],
                "superclass_labels": [
                    superclass
                    for superclass in dag.nodes_in_layer("superclasses")
                    if smiles_labels[superclass] == 1
                ],
                "class_labels": [
                    klass
                    for klass in dag.nodes_in_layer("classes")
                    if smiles_labels[klass] == 1
                ],
            }
            for smiles, smiles_labels in tqdm(
                harmonized_df.iterrows(),
                total=harmonized_df.shape[0],
                desc="Generating harmonized dataset",
            )
        ],
        "../hammer/datasets/npc/npc-harmonized.json.xz",
    )


if __name__ == "__main__":
    harmonize_npc()
