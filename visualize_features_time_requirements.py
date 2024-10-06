"""Script to create barplots visualizing the performance of different feature sets."""

import os
import pandas as pd
from barplots import barplots


def visualize_features_time_requirements():
    """Visualize the performance of different feature sets."""
    if not os.path.exists("features_time_requirements.csv"):
        raise FileNotFoundError(
            "features_time_requirements.csv not found. Run feature_set_selection.py first."
        )

    performance = pd.read_csv("features_time_requirements.csv")

    performance["parameter"] = (
        performance["parameter"]
        .str.replace("include_", "")
        .str.replace("_fingerprint", " ")
        .str.replace("_", " ")
    )

    barplots(
        performance,
        groupby=["parameter"],
        orientation="horizontal",
        height=6,
        show_last_level_as_legend=False,
        legend_position="upper left",
    )


if __name__ == "__main__":
    visualize_features_time_requirements()
