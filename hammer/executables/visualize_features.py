"""Executable to visualize the features of the dataset."""

import os
from collections import Counter
from typing import Type, List, Dict
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rdkit.Chem.rdchem import Mol
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from matplotlib.colors import TABLEAU_COLORS
from hammer.datasets import Dataset
from hammer.dags import DAG
from hammer.molecular_features import FeatureInterface
from hammer.feature_settings import FeatureSettings
from hammer.augmentation_settings import AugmentationSettings
from hammer.executables.argument_parser_utilities import (
    add_augmentation_settings_arguments,
    build_augmentation_settings_from_namespace,
    add_features_arguments,
    build_features_settings_from_namespace,
    add_shared_arguments,
    add_dataset_arguments,
    build_dataset_from_namespace,
)
from hammer.utils import smiles_to_molecules, into_canonical_smiles


TABLEAU_COLORS: List[str] = list(TABLEAU_COLORS.values())


def _visualize_feature(
    molecules: List[Mol],
    labels: Dict[str, np.ndarray],
    most_common: Dict[str, np.ndarray],
    top: Dict[str, List[str]],
    feature: Type[FeatureInterface],
    arguments: Namespace,
):
    """Visualize a feature."""
    assert isinstance(molecules, list)
    assert isinstance(labels, dict)
    assert isinstance(most_common, dict)
    assert isinstance(top, dict)
    assert issubclass(feature.__class__, FeatureInterface)
    assert isinstance(arguments, Namespace)

    x: np.ndarray = feature.transform_molecules(molecules)

    if not feature.__class__.is_binary():
        x = RobustScaler().fit_transform(x)

    if np.isnan(x).any():
        return

    try:
        from MulticoreTSNE import MulticoreTSNE # pylint: disable=import-outside-toplevel
        tsne = MulticoreTSNE(
            n_components=2,
            n_jobs=arguments.n_jobs,
            verbose=1,
            n_iter_early_exag=5 if arguments.smoke_test else 250,
            n_iter=10 if arguments.smoke_test else 1000,
        )
    except ImportError:
        from sklearn.manifold import TSNE # pylint: disable=import-outside-toplevel
        tsne = TSNE(
            n_components=2,
            n_jobs=arguments.n_jobs,
            verbose=1,
            n_iter=10 if arguments.smoke_test else 1000,
        )

    pca = PCA(n_components=50)

    if x.shape[1] > 50:
        # First, we reduce the dimensionality of the features to 50
        x_reduced: np.ndarray = pca.fit_transform(x)
    else:
        x_reduced: np.ndarray = x

    # We compute the 2d version using PCA
    x_pca: np.ndarray = PCA(n_components=2).fit_transform(x)

    # Next, we finish the reduction to 2 dimensions with t-SNE
    x_tsne: np.ndarray = tsne.fit_transform(x_reduced)

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(21, 14), dpi=200)

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

    visualize_features_parser = add_dataset_arguments(
        add_features_arguments(
            add_shared_arguments(
                add_augmentation_settings_arguments(visualize_features_parser)
            )
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

    dataset: Type[Dataset] = build_dataset_from_namespace(namespace=arguments)
    dag: Type[DAG] = dataset.layered_dag()
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
        layer: List[str] = dag.get_layer(label_name)
        for smile_labels in label_values:
            for i, smile_label in enumerate(smile_labels):
                if smile_label == 1:
                    counters[label_name].update([layer[i]])

    # Since we can't possibly show all of the superclasses and classes,
    # we will only show the top 'number of colors' of each. Some samples will have multiple
    # pathways, superclasses and classes, so we will only show the most common ones.
    number_of_colors = len(TABLEAU_COLORS)

    # We determine the top 'number_of_colors - 1' most common labels

    top_labels: Dict[str, List[str]] = {}
    for layer_name in dag.get_layer_names():
        top_labels[layer_name] = sorted(
            counters[layer_name], key=lambda x: counters[layer_name][x], reverse=True
        )[: number_of_colors - 1]

    # Plus one for "Other"
    for layer_name in dag.get_layer_names():
        if len(counters[layer_name]) > number_of_colors - 1:
            top_labels[layer_name].append("Other")

    most_common_labels: Dict[str, List[str]] = {
        layer_name: [] for layer_name in dag.get_layer_names()
    }
    for layer_name in dag.get_layer_names():
        for one_hot_encoded_y in labels[layer_name]:
            most_common_label = None
            most_common_count = 0
            for bit, label_name in zip(one_hot_encoded_y, dag.get_layer(layer_name)):
                if bit == 1 and counters[layer_name][label_name] > most_common_count:
                    most_common_label = label_name
                    most_common_count = counters[layer_name][label_name]

            assert most_common_label is not None

            try:
                most_common_index = top_labels[layer_name].index(most_common_label)
            except ValueError:
                most_common_index = number_of_colors - 1
            most_common_labels[layer_name].append(most_common_index)

    # We convert the most common labels to numpy arrays
    most_common_labels = {
        key: np.array(value) for key, value in most_common_labels.items()
    }

    canonical_smiles: List[str] = into_canonical_smiles(
        smiles, n_jobs=arguments.n_jobs, verbose=arguments.verbose
    )
    molecules: List[Mol] = smiles_to_molecules(
        canonical_smiles, verbose=arguments.verbose, n_jobs=arguments.n_jobs
    )

    for feature_class in tqdm(
        feature_settings.iter_features(),
        desc="Visualizing features",
        total=feature_settings.number_of_features(),
        disable=not arguments.verbose,
        leave=False,
        dynamic_ncols=True,
    ):
        _visualize_feature(
            molecules,
            labels,
            most_common_labels,
            top_labels,
            feature_class(
                verbose=arguments.verbose,
                n_jobs=arguments.n_jobs,
            ),
            arguments,
        )
