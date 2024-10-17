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

from typing import Optional, Dict, Set
import pandas as pd
import compress_json
from hammer.datasets.labeled_smiles import LabeledSMILES
from hammer.datasets import NPCDataset, NPCScrapedDataset

# ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"


def try_to_harmonize_with_rules(
    original: LabeledSMILES,
    scraped: LabeledSMILES,
) -> Optional[LabeledSMILES]:
    """Try to harmonize the original and scraped entries."""

    # If the original entry is compatible with the scraped entry, we consider the scraped entry
    # as the ground truth.

    first_discordant_layer: Optional[str] = original.first_discordant_layer(
        scraped, ["pathways", "classes", "superclasses"]
    )

    # If the discordance is at the pathway level, there is not much we can do here.
    if first_discordant_layer == "pathways":
        return None

    if first_discordant_layer == "classes":
        # If the scraped entry contains "nucleos(t)ide", the difference is related
        # to the class being renamed. We consider the original entry as the ground truth.
        if any("nucleos(t)ide" in klass for klass in scraped.labels["classes"]):
            return original

    if first_discordant_layer == "superclasses":
        if (
            any(
                "nucleoside" in superclass
                for superclass in scraped.labels["superclasses"]
            )
            and any(
                "nucleotide" in superclass
                for superclass in original.labels["superclasses"]
            )
            and any("nucleotide" in klass for klass in scraped.labels["classes"])
        ):
            return original

    return None


def try_to_harmonize_with_manual_correction(
    original: LabeledSMILES,
    scraped: LabeledSMILES,
    manual_corrections: Dict[str, pd.DataFrame],
) -> Optional[LabeledSMILES]:
    """Try to harmonize the original and scraped entries with a manual correction."""
    for layer_name in ["pathways", "classes", "superclasses"]:
        correction: pd.DataFrame = manual_corrections[layer_name]

        if original.smiles not in correction.smiles.values:
            continue

        original_row = correction[correction.smiles == original.smiles].iloc[0]

        # If there is no correction for the layer, we can't determine the correct prediction.
        if original_row.correct_option == "?":
            return None

        if original_row.correct_option == "original":
            return original

        if original_row.correct_option == "scraped":
            return scraped

        if original_row.correct_option == "both":
            return original.merge_labels(scraped)

        raise NotImplementedError(
            f"Unknown correction option: {original_row.correct_option}"
        )

    return None


def harmonize_npc():
    """Tries to harmonize the NPC dataset."""

    original = NPCDataset()
    scraped = NPCScrapedDataset()

    original: Dict[str, LabeledSMILES] = {
        entry.smiles: entry for entry in original.iter_labeled_smiles()
    }
    scraped: Dict[str, LabeledSMILES] = {
        entry.smiles: entry for entry in scraped.iter_labeled_smiles()
    }

    manual_corrections: Dict[str, pd.DataFrame] = {
        "pathways": pd.read_csv("../divergent_npc_entries/divergent_pathways.csv"),
        "classes": pd.read_csv("../divergent_npc_entries/divergent_classes.csv"),
        "superclasses": pd.read_csv(
            "../divergent_npc_entries/divergent_superclasses.csv"
        ),
    }

    harmonized: Dict[str, LabeledSMILES] = {}
    discarded_smiles: Set[str] = set()
    discarded_smiles_divergence_type: Dict[str, int] = {}
    harmonization_tecniques_counts: Dict[str, int] = {
        "scraped_is_empty": 0,
        "no_divergence": 0,
        "rules": 0,
        "manual_corrections": 0,
        "scraped_is_complete": 0,
    }

    for smiles, original_entry in original.items():
        scraped_entry = scraped.get(smiles)
        if scraped_entry is None or scraped_entry.has_missing_labels(
            ["pathways", "classes", "superclasses"]
        ):
            harmonized[smiles] = original_entry
            harmonization_tecniques_counts["scraped_is_empty"] += 1
            continue

        first_discordant_layer: Optional[str] = original_entry.first_discordant_layer(
            scraped_entry, ["pathways", "classes", "superclasses"]
        )

        if first_discordant_layer is None:
            harmonized[smiles] = scraped_entry
            harmonization_tecniques_counts["no_divergence"] += 1
            continue

        harmonized_entry = try_to_harmonize_with_rules(original_entry, scraped_entry)

        if harmonized_entry is not None:
            harmonized[smiles] = harmonized_entry
            harmonization_tecniques_counts["rules"] += 1
            continue

        harmonized_entry = try_to_harmonize_with_manual_correction(
            original_entry, scraped_entry, manual_corrections
        )

        if harmonized_entry is not None:
            harmonized[smiles] = harmonized_entry
            harmonization_tecniques_counts["manual_corrections"] += 1
            continue

        discarded_smiles.add(smiles)
        discarded_smiles_divergence_type[first_discordant_layer] = (
            discarded_smiles_divergence_type.get(first_discordant_layer, 0) + 1
        )

    for smiles, scraped_entry in scraped.items():
        if smiles in original:
            continue

        # If the SMILES only appears in the scraped dataset,
        # we consider the scraped prediction as the ground truth
        # if it is complete.
        if not scraped_entry.has_missing_labels(
            ["pathways", "classes", "superclasses"]
        ):
            harmonized[smiles] = scraped_entry
            harmonization_tecniques_counts["scraped_is_complete"] += 1
            continue

        discarded_smiles.add(smiles)

    print(f"{BOLD}{RED}Discarded:{RESET} {len(discarded_smiles)} entries")

    for divergence_type, count in discarded_smiles_divergence_type.items():
        print(f"{YELLOW}└── {divergence_type}:{RESET} {count}")

    print(f"{BOLD}{GREEN}Harmonized:{RESET} {len(harmonized)} entries")

    for technique, count in harmonization_tecniques_counts.items():
        print(f"{BLUE}└── {technique}:{RESET} {count}")

    compress_json.dump(
        [harmonized_entry.to_dict() for harmonized_entry in harmonized.values()],
        "../hammer/datasets/npc/npc-harmonized.json.xz",
    )


if __name__ == "__main__":
    harmonize_npc()
