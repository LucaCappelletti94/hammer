"""Executor to visualize the features."""

import os
from collections import Counter
from multiprocessing import cpu_count
import silence_tensorflow.auto  # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE
from barplots.barplot import EXTENDED_COLORS
from np_classifier.training import Dataset


def visualize():
    """Train the model."""
    dataset = Dataset(
        include_atom_pair_fingerprint=True,
        include_maccs_fingerprint=True,
        include_morgan_fingerprint=True,
        include_rdkit_fingerprint=True,
        include_avalon_fingerprint=True,
        include_descriptors=True,
        include_map4_fingerprint=True,
        include_topological_torsion_fingerprint=True,
        include_skfp_autocorr_fingerprint=True,
        include_skfp_ecfp_fingerprint=True,
        include_skfp_erg_fingerprint=True,
        include_skfp_estate_fingerprint=True,
        include_skfp_functional_groups_fingerprint=True,
        include_skfp_ghose_crippen_fingerprint=True,
        include_skfp_klekota_roth_fingerprint=True,
        include_skfp_laggner_fingerprint=True,
        include_skfp_layered_fingerprint=True,
        include_skfp_lingo_fingerprint=True,
        include_skfp_map_fingerprint=True,
        include_skfp_mhfp_fingerprint=True,
        include_skfp_mqns_fingerprint=True,
        include_skfp_pattern_fingerprint=True,
        include_skfp_pubchem_fingerprint=True,
        include_skfp_secfp_fingerprint=True,
        include_skfp_vsa_fingerprint=True,
    )
    # We compute the features without augmentation
    _scalers, (train_x, train_y), _ = dataset.primary_split(augment=False)

    assert len(train_x) > 0
    assert len(train_y) > 0

    # Since we can't possibly show all of the superclasses and classes,
    # we will only show the top 'number of colors' of each. Some samples will have multiple
    # pathways, superclasses and classes, so we will only show the most common ones.
    colors = EXTENDED_COLORS[:16]
    number_of_colors = len(colors)

    counters = {}

    for key, labels in train_y.items():
        counters[key] = Counter()
        for label in labels:
            for i, bit in enumerate(label):
                if bit == 1:
                    if key == "pathway":
                        counters[key].update([dataset.pathway_names[i]])
                    elif key == "superclass":
                        counters[key].update([dataset.superclass_names[i]])
                    elif key == "class":
                        counters[key].update([dataset.class_names[i]])

    # We determine the top 'number_of_colors - 1' most common pathways, superclasses and classes
    top_pathways = sorted(
        counters["pathway"], key=lambda x: counters["pathway"][x], reverse=True
    )[: number_of_colors - 1]
    top_superclasses = sorted(
        counters["superclass"], key=lambda x: counters["superclass"][x], reverse=True
    )[: number_of_colors - 1]
    top_classes = sorted(
        counters["class"], key=lambda x: counters["class"][x], reverse=True
    )[: number_of_colors - 1]

    # Plus one for "Other"
    if len(counters["pathway"]) > number_of_colors - 1:
        top_pathways.append("Other")
    if len(counters["superclass"]) > number_of_colors - 1:
        top_superclasses.append("Other")
    if len(counters["class"]) > number_of_colors - 1:
        top_classes.append("Other")

    # We determine the most common pathways, superclasses and classes for each sample.
    # When the most common entry of a sample is not in the top ones, we will replace it with "Other".
    most_common_pathways = []
    most_common_superclasses = []
    most_common_classes = []
    for labels, counter, label_names, most_common, top in [
        (
            train_y["pathway"],
            counters["pathway"],
            dataset.pathway_names,
            most_common_pathways,
            top_pathways,
        ),
        (
            train_y["superclass"],
            counters["superclass"],
            dataset.superclass_names,
            most_common_superclasses,
            top_superclasses,
        ),
        (
            train_y["class"],
            counters["class"],
            dataset.class_names,
            most_common_classes,
            top_classes,
        ),
    ]:
        for one_hot_encoded_y in labels:
            most_common_label = None
            most_common_count = 0
            for bit, label_name in zip(one_hot_encoded_y, label_names):
                if bit == 1 and counter[label_name] > most_common_count:
                    most_common_label = label_name
                    most_common_count = counter[label_name]

            assert most_common_label is not None

            try:
                most_common_index = top.index(most_common_label)
            except ValueError:
                most_common_index = number_of_colors - 1
            most_common.append(most_common_index)

    most_common_pathways = np.array(most_common_pathways)
    most_common_superclasses = np.array(most_common_superclasses)
    most_common_classes = np.array(most_common_classes)

    train_x["composite"] = np.concatenate(
        [
            train_x[feature_set_name]
            for feature_set_name in train_x
            if not np.isnan(train_x[feature_set_name]).any()
        ],
        axis=1,
    )

    for feature_set_name, features in tqdm(
        train_x.items(),
        desc="Visualizing features",
        total=len(train_x),
        unit="feature set",
        leave=False,
        dynamic_ncols=True,
    ):
        path = f"data_visualizations/{feature_set_name}.png"

        if os.path.exists(path):
            continue

        if np.isnan(features).any():
            print(f"Skipping {feature_set_name} due to NaN values")
            continue

        pca = PCA(n_components=50)
        tsne = MulticoreTSNE(n_components=2, n_jobs=cpu_count(), verbose=0)

        if features.shape[1] > 50:
            # First, we reduce the dimensionality of the features to 50
            pca_features = pca.fit_transform(features)
        else:
            pca_features = features
        # Next, we finish the reduction to 2 dimensions with t-SNE
        tsne_features = tsne.fit_transform(pca_features)

        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(30, 10), dpi=300)

        for i, (label_name, indices, top_labels) in enumerate(
            [
                ("Pathway", most_common_pathways, top_pathways),
                ("Superclass", most_common_superclasses, top_superclasses),
                ("Class", most_common_classes, top_classes),
            ]
        ):
            ax[i].scatter(
                tsne_features[:, 0],
                tsne_features[:, 1],
                c=[colors[index] for index in indices],
                marker=".",
                alpha=0.5,
            )
            ax[i].set_title(f"{feature_set_name} - {label_name}")
            ax[i].set_xlabel("t-SNE 1")
            ax[i].set_ylabel("t-SNE 2")

            # We populate and display the legend
            handles = []
            labels = []
            for j, label in enumerate(top_labels):
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[j],
                        label=label,
                    )
                )
                labels.append(label)
            ax[i].legend(
                handles,
                labels,
                loc="upper right",
                prop={'size': 8}
            )

        plt.tight_layout()

        os.makedirs("data_visualizations", exist_ok=True)
        fig.savefig(path)
        plt.close()


if __name__ == "__main__":
    visualize()
