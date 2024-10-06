"""Script to create barplots visualizing the performance of different feature sets."""

import os
import pandas as pd
import compress_json
from barplots import barplots


def visualize_feature_sets_performance():
    """Visualize the performance of different feature sets."""
    if not os.path.exists("feature_sets_performance.csv"):
        raise FileNotFoundError(
            "feature_sets_performance.csv not found. Run feature_set_selection.py first."
        )

    performance = pd.concat([
        pd.read_csv("feature_sets_performance.csv"),
        pd.DataFrame(compress_json.load("feature_sets_performance.json"))
    ])

    performance["feature_set"] = performance["feature_set"].str.replace("fingerprint", "")

    barplots(
        performance,
        path="barplots/{feature}_feature_sets.png",
        groupby=["set", "feature_set"],
        show_last_level_as_legend=False,
        subplots=True,
        unique_minor_labels=False,
        orientation="horizontal",
        height=6,
        legend_position="lower left",
    )


if __name__ == "__main__":
    visualize_feature_sets_performance()
