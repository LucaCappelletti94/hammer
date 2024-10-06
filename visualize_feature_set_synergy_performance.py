"""Script to create barplots visualizing the performance of different feature sets."""

import os
import pandas as pd
from barplots import barplots
import compress_json


def visualize_feature_sets_performance():
    """Visualize the performance of different feature sets."""
    if not os.path.exists("feature_sets_performance.csv"):
        raise FileNotFoundError(
            "feature_sets_performance.csv not found. Run feature_set_selection.py first."
        )

    performance = pd.read_csv("feature_sets_performance.csv")
    performance["first_feature_set"] = performance["feature_set"]
    performance["second_feature_set"] = performance["feature_set"]

    synergy_performance = pd.DataFrame(
        compress_json.load("feature_set_synergy_performance.json")
    )

    # We make sure that the simmetric variant are present in the dataframe, or we
    # add them with the same value
    symmetric_variants = []
    for _, row in synergy_performance.iterrows():
        symmetric_variant = row.copy()
        symmetric_variant["first_feature_set"] = row["second_feature_set"]
        symmetric_variant["second_feature_set"] = row["first_feature_set"]
        symmetric_variants.append(symmetric_variant)

    symmetric_variants = pd.DataFrame(symmetric_variants)

    performance = pd.concat(
        [performance, symmetric_variants, synergy_performance], ignore_index=True
    )

    barplots(
        performance,
        groupby=["first_feature_set", "set", "second_feature_set"],
        orientation="horizontal",
        path="barplots/{feature}_synergy.png",
        height=6,
        plots_per_row=6,
        subplots=True,
        show_last_level_as_legend=False,
        unique_minor_labels=False,
    )


if __name__ == "__main__":
    visualize_feature_sets_performance()
