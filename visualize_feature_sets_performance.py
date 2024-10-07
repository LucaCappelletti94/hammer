"""Script to create barplots visualizing the performance of different feature sets."""

import os
import pandas as pd
from barplots import barplots


def sort_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the bars in the barplot."""

    df_copy = df.reset_index()

    df["sort_params"] = [
        param.replace("skfp", "").lower().strip() for param in df_copy["feature_set"]
    ]
    df = df.sort_values(by=["sort_params"])
    df = df.drop(columns=["sort_params"])

    return df

def visualize_feature_sets_performance():
    """Visualize the performance of different feature sets."""
    if not os.path.exists("feature_sets_performance.csv"):
        raise FileNotFoundError(
            "feature_sets_performance.csv not found. Run feature_set_selection.py first."
        )

    performance = pd.read_csv("feature_sets_performance.csv")
    performance_4096 = pd.read_csv("feature_sets_performance_4096.csv")
    performance_radius2 = pd.read_csv("feature_sets_performance_radius2.csv")

    performance["feature_set"] = performance["feature_set"].str.replace("fingerprint", "")
    performance_4096["feature_set"] = performance_4096["feature_set"].str.replace("fingerprint", "")
    performance_radius2["feature_set"] = performance_radius2["feature_set"].str.replace("fingerprint", "")

    performance_4096["feature_set"] = [
        f"{feature_set} (4096 bits)" for feature_set in performance_4096["feature_set"]
    ]
    performance_radius2["feature_set"] = [
        f"{feature_set} (radius 2)" for feature_set in performance_radius2["feature_set"]
    ]

    performance = pd.concat([performance, performance_4096, performance_radius2])

    barplots(
        performance,
        path="barplots/{feature}_feature_sets.png",
        groupby=["set", "feature_set"],
        show_last_level_as_legend=False,
        subplots=True,
        unique_minor_labels=False,
        orientation="horizontal",
        height=6,
        sort_bars=sort_bars,
        legend_position="lower left",
    )


if __name__ == "__main__":
    visualize_feature_sets_performance()
