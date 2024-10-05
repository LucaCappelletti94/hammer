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
    
    synergy_performance = pd.DataFrame(compress_json.load("feature_set_synergy_performance.json"))

    performance = pd.concat([performance, synergy_performance], ignore_index=True)

    barplots(
        performance,
        groupby=["first_feature_set", "set", "second_feature_set"],
        orientation="horizontal",
        height=4,
        subplots=True,
        legend_position="lower left",
    )


if __name__ == "__main__":
    visualize_feature_sets_performance()
