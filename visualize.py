"""Executor to visualize the features."""

import silence_tensorflow.auto  # pylint: disable=unused-import
import os
import pandas as pd
import numpy as np
import compress_json
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from np_classifier.training import Trainer, Dataset


def visualize():
    """Train the model."""
    dataset = Dataset()
    (train_x, train_y), _ = dataset.primary_split()

    # Since we can't possibly show all of the superclasses and classes,
    # we will only show the top 'number of colors' of each. Some samples will have multiple
    # pathways, superclasses and classes, so we will only show the most common ones.

    colors = list(TABLEAU_COLORS.values())
    number_of_colors = len(colors)

    pathway_counts = dataset.training_pathway_counts()
    superclass_counts = dataset.training_superclass_counts()
    class_counts = dataset.training_class_counts()

    # We determine the top 'number_of_colors - 1' most common pathways, superclasses and classes
    top_pathways = sorted(
        pathway_counts, key=lambda x: pathway_counts[x], reverse=True
    )[: number_of_colors - 1]
    top_superclasses = sorted(
        superclass_counts, key=lambda x: superclass_counts[x], reverse=True
    )[: number_of_colors - 1]
    top_classes = sorted(class_counts, key=lambda x: class_counts[x], reverse=True)[
        : number_of_colors - 1
    ]

    # Plus one for "Other"
    top_pathways.append("Other")
    top_superclasses.append("Other")
    top_classes.append("Other")

    # We determine the most common pathways, superclasses and classes for each sample.
    # When the most common entry of a sample is not in the top 11, we will replace it with "Other".
    most_common_pathways = []
    most_common_superclasses = []
    most_common_classes = []
    for molecule in tqdm(
        dataset.training_molecules,
        desc="Determining most common labels",
        total=len(dataset.training_molecules),
        unit="molecule",
        leave=False,
        dynamic_ncols=True,
    ):
        most_common_pathway = molecule.most_common_pathway_label_name(pathway_counts)
        most_common_superclass = molecule.most_common_superclass_label_name(
            superclass_counts
        )
        most_common_class = molecule.most_common_class_label_name(class_counts)

        try:
            most_common_pathway_index = top_pathways.index(most_common_pathway)
        except ValueError:
            most_common_pathway_index = number_of_colors - 1
        try:
            most_common_superclass_index = top_superclasses.index(
                most_common_superclass
            )
        except ValueError:
            most_common_superclass_index = number_of_colors - 1
        try:
            most_common_class_index = top_classes.index(most_common_class)
        except ValueError:
            most_common_class_index = number_of_colors - 1

        most_common_pathways.append(most_common_pathway_index)
        most_common_superclasses.append(most_common_superclass_index)
        most_common_classes.append(most_common_class_index)

    most_common_pathways = np.array(most_common_pathways)
    most_common_superclasses = np.array(most_common_superclasses)
    most_common_classes = np.array(most_common_classes)

    train_x["composite"] = np.concatenate(
        [train_x[feature_set_name] for feature_set_name in train_x], axis=1
    )

    for feature_set_name, features in tqdm(
        train_x.items(),
        desc="Visualizing features",
        total=len(train_x),
        unit="feature set",
        leave=False,
        dynamic_ncols=True,
    ):
        pca = PCA(n_components=50)
        tsne = MulticoreTSNE(n_components=2, n_jobs=cpu_count(), verbose=1)

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
            ax[i].legend(handles, labels, loc="upper right")

        plt.tight_layout()

        os.makedirs("data_visualizations", exist_ok=True)
        plt.savefig(f"data_visualizations/{feature_set_name}.png")
        plt.close()


if __name__ == "__main__":
    visualize()
