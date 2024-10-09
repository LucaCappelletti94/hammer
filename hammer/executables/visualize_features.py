"""Executable to visualize the features of the dataset."""

import os
from collections import Counter
from typing import Type, List, Dict
from argparse import Namespace
import silence_tensorflow.auto  # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from MulticoreTSNE import MulticoreTSNE
from matplotlib.colors import TABLEAU_COLORS
from hammer.training import (
    Dataset,
    AugmentationSettings,
    FeatureSettings,
    FeatureInterface,
)
from hammer.executables.argument_parser_utilities import (
    add_augmentation_settings_arguments,
    build_augmentation_settings_from_namespace,
    add_features_arguments,
    build_features_settings_from_namespace,
    add_shared_arguments,
)


TABLEAU_COLORS: List[str] = list(TABLEAU_COLORS.values())


def _visualize_feature(
    smiles: List[str],
    labels: Dict[str, np.ndarray],
    most_common: Dict[str, np.ndarray],
    top: Dict[str, List[str]],
    feature: Type[FeatureInterface],
    arguments: Namespace,
):
    """Visualize a feature."""
    assert isinstance(smiles, list)
    assert isinstance(labels, dict)
    assert isinstance(most_common, dict)
    assert isinstance(top, dict)
    assert issubclass(feature.__class__, FeatureInterface)
    assert isinstance(arguments, Namespace)

    x: np.ndarray = feature.transform_molecules(smiles)

    if not feature.__class__.is_binary():
        x = RobustScaler().fit_transform(x)

    if np.isnan(x).any():
        print(f"Skipping {feature.name()} due to NaN values")
        return

    pca = PCA(n_components=50)
    tsne = MulticoreTSNE(
        n_components=2,
        n_jobs=arguments.n_jobs,
        verbose=1,
        n_iter_early_exag=5 if arguments.smoke_test else 250,
        n_iter=10 if arguments.smoke_test else 1000,
    )

    if x.shape[1] > 50:
        # First, we reduce the dimensionality of the features to 50
        x_reduced: np.ndarray = pca.fit_transform(x)
    else:
        x_reduced: np.ndarray = x

    # We compute the 2d version using PCA
    x_pca: np.ndarray = PCA(n_components=2).fit_transform(x)

    # Next, we finish the reduction to 2 dimensions with t-SNE
    x_tsne: np.ndarray = tsne.fit_transform(x_reduced)

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(21, 14), dpi=120)

    for i, (decomposition, decomposition_name) in enumerate(
        [(x_pca, "PCA"), (x_tsne, "t-SNE")]
    ):
        for j, (label_name, common) in enumerate(most_common.items()):
            top_labels = top[label_name]
            ax[i, j].scatter(
                decomposition[:, 0],
                decomposition[:, 1],
                c=[TABLEAU_COLORS[color_index] for color_index in common],
                marker=".",
                alpha=0.5,
            )
            ax[i, j].set_title(f"{feature.name()} - {label_name.capitalize()}")
            ax[i, j].set_xlabel(f"{decomposition_name} 1")
            if j == 0:
                ax[i, j].set_ylabel(f"{decomposition_name} 2")

            # We populate and display the legend
            handles = []
            labels = []
            for k, label in enumerate(top_labels):
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=TABLEAU_COLORS[k],
                        label=label,
                    )
                )
                labels.append(label)
            ax[i, j].legend(handles, labels, ncols=3, prop={"size": 8})

    plt.tight_layout()

    os.makedirs(arguments.output_directory, exist_ok=True)
    fig.savefig(
        os.path.join(
            arguments.output_directory, f"{feature.name()}.{arguments.image_format}"
        )
    )
    plt.close()


def add_visualize_features_subcommand(sub_parser_action: "SubParserAction"):
    """Add the visualize features sub-command to the parser."""
    visualize_features_parser = sub_parser_action.add_parser(
        "visualize", help="Visualize the features of the dataset."
    )

    visualize_features_parser = add_features_arguments(
        add_shared_arguments(
            add_augmentation_settings_arguments(visualize_features_parser)
        )
    )

    # We add an additional argument for the directory where to store
    # the visualizations.
    visualize_features_parser.add_argument(
        "--output-directory",
        type=str,
        default="data_visualizations",
        help="The directory where to store the visualizations.",
    )

    # And the image format, which can be either 'png', 'jpg', 'pdf' or 'jpeg'.
    visualize_features_parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "pdf", "jpeg"],
        help="The format of the images.",
    )
    visualize_features_parser.set_defaults(func=visualize)


def visualize(arguments: Namespace):
    """Train the model."""

    if arguments.smoke_test:
        maximal_number_of_molecules = 2000
    else:
        maximal_number_of_molecules = None

    dataset = Dataset(
        maximal_number_of_molecules=maximal_number_of_molecules,
        verbose=arguments.verbose,
    )
    smiles, labels = dataset.all_smiles()

    # We construct the augmentation strategies from the argument parser.
    augmentation_settings: AugmentationSettings = (
        build_augmentation_settings_from_namespace(arguments)
    )

    smiles, labels = augmentation_settings.augment(smiles, labels)

    # We construct the feature settings from the argument parser.
    feature_settings: FeatureSettings = build_features_settings_from_namespace(
        arguments
    )

    if not feature_settings.includes_features():
        feature_settings = FeatureSettings().include_all()

    counters: Dict[str, Counter] = {}

    for label_name, label_values in labels.items():
        counters[label_name] = Counter()
        for smile_labels in label_values:
            for i, smile_label in enumerate(smile_labels):
                if smile_label == 1:
                    if label_name == "pathway":
                        counters[label_name].update([dataset.pathway_names[i]])
                    elif label_name == "superclass":
                        counters[label_name].update([dataset.superclass_names[i]])
                    elif label_name == "class":
                        counters[label_name].update([dataset.class_names[i]])

    # Since we can't possibly show all of the superclasses and classes,
    # we will only show the top 'number of colors' of each. Some samples will have multiple
    # pathways, superclasses and classes, so we will only show the most common ones.
    number_of_colors = len(TABLEAU_COLORS)

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

    top: Dict[str, List[str]] = {
        "pathway": top_pathways,
        "superclass": top_superclasses,
        "class": top_classes,
    }

    most_common_pathways = []
    most_common_superclasses = []
    most_common_classes = []
    for sub_labels, counter, unique_label_names, most_common, top_labels in [
        (
            labels["pathway"],
            counters["pathway"],
            dataset.pathway_names,
            most_common_pathways,
            top_pathways,
        ),
        (
            labels["superclass"],
            counters["superclass"],
            dataset.superclass_names,
            most_common_superclasses,
            top_superclasses,
        ),
        (
            labels["class"],
            counters["class"],
            dataset.class_names,
            most_common_classes,
            top_classes,
        ),
    ]:
        for one_hot_encoded_y in sub_labels:
            most_common_label = None
            most_common_count = 0
            for bit, label_name in zip(one_hot_encoded_y, unique_label_names):
                if bit == 1 and counter[label_name] > most_common_count:
                    most_common_label = label_name
                    most_common_count = counter[label_name]

            assert most_common_label is not None

            try:
                most_common_index = top_labels.index(most_common_label)
            except ValueError:
                most_common_index = number_of_colors - 1
            most_common.append(most_common_index)

    most_common_pathways = np.array(most_common_pathways)
    most_common_superclasses = np.array(most_common_superclasses)
    most_common_classes = np.array(most_common_classes)

    most_common: Dict[str, np.ndarray] = {
        "pathway": most_common_pathways,
        "superclass": most_common_superclasses,
        "class": most_common_classes,
    }

    for feature_class in tqdm(
        feature_settings.iter_features(),
        desc="Visualizing features",
        total=feature_settings.number_of_features(),
        disable=not arguments.verbose,
        leave=False,
        dynamic_ncols=True,
    ):
        _visualize_feature(
            smiles,
            labels,
            most_common,
            top,
            feature_class(
                verbose=arguments.verbose,
                n_jobs=arguments.n_jobs,
            ),
            arguments,
        )
