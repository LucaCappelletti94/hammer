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
    mean_performance = performance.groupby(["feature_set", "set"]).mean().reset_index()

    performance["first_feature_set"] = performance["feature_set"]
    performance["second_feature_set"] = performance["feature_set"]

    synergy_performance = pd.DataFrame(
        compress_json.load("feature_set_synergy_performance.json")
    )

    # Some features we have just computed in the synergy performance
    # have been removed, so if they do not appear in the performance
    # CSV, we remove them from the synergy performance dataframe.
    synergy_performance_filtered = []
    for _, row in synergy_performance.iterrows():
        if (
            row["first_feature_set"] in performance["feature_set"].values
            and row["second_feature_set"] in performance["feature_set"].values
        ):
            synergy_performance_filtered.append(row)
    synergy_performance = pd.DataFrame(synergy_performance_filtered)

    # We make sure that the simmetric variant are present in the dataframe, or we
    # add them with the same value
    symmetric_variants = []
    for _, row in synergy_performance.iterrows():
        symmetric_variant = row.copy()
        symmetric_variant["first_feature_set"] = row["second_feature_set"]
        symmetric_variant["second_feature_set"] = row["first_feature_set"]
        symmetric_variants.append(symmetric_variant)

    symmetric_variants = pd.DataFrame(symmetric_variants)

    synergy_performance = pd.concat(
        [performance, symmetric_variants, synergy_performance], ignore_index=True
    )

    # Next, we compute for each feature (average_precision and accuracy)
    # the improvement rate over the performance of the feature with the
    # best performance out of the feature tuples.

    improvements = []
    for _, row in synergy_performance.iterrows():
        for evaluation_set in ["train", "valid"]:
            
            first_feature_set = row["first_feature_set"]
            second_feature_set = row["second_feature_set"]
            first_feature_set = first_feature_set.replace("fingerprint", "")
            second_feature_set = second_feature_set.replace("fingerprint", "")

            this_improvement = {
                "first_feature_set": first_feature_set,
                "second_feature_set": second_feature_set,
                "set": evaluation_set,
            }
            for metric in ["average_precision", "accuracy"]:

                best_performance = mean_performance[
                    (
                        (mean_performance["feature_set"] == row["first_feature_set"])
                        | (mean_performance["feature_set"] == row["second_feature_set"])
                    )
                    & (mean_performance["set"] == evaluation_set)
                ][metric].max()

                if best_performance is None:
                    raise ValueError(
                        f"Could not find best performance for {row['first_feature_set']} and {row['second_feature_set']}"
                    )

                improvement = (
                    row[metric] - best_performance
                ) / best_performance
                this_improvement[f"{metric}_change"] = improvement
                this_improvement[metric] = row[metric]
            improvements.append(this_improvement)

    improvements = pd.DataFrame(improvements)
    # synergy_performance.drop(columns=["average_precision", "accuracy"], inplace=True)

    improvements = improvements[improvements["set"] == "valid"]

    barplots(
        improvements,
        groupby=["first_feature_set", "second_feature_set"],
        orientation="horizontal",
        path="barplots/{feature}_synergy.png",
        height=6,
        plots_per_row=7,
        subplots=True,
        show_standard_deviation=False,
        show_last_level_as_legend=False,
        unique_major_labels=False,
        unique_minor_labels=False,
    )


if __name__ == "__main__":
    visualize_feature_sets_performance()
