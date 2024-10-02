"""Script to create barplots visualizing the performance of different feature sets."""

import os
import pandas as pd
from barplots import barplots


def visualize_feature_sets_performance():
    """Visualize the performance of different feature sets."""
    if not os.path.exists("feature_sets_performance.csv"):
        raise FileNotFoundError(
            "feature_sets_performance.csv not found. Run feature_set_selection.py first."
        )

    performance = pd.read_csv("feature_sets_performance.csv")

    barplots(
        performance,
        groupby=["set", "feature_set"],
        orientation="horizontal",
        height=4,
        legend_position="upper left",
    )


if __name__ == "__main__":
    visualize_feature_sets_performance()
