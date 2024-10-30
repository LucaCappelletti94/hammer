"""Executable to visualize the features of the dataset."""

import os
from typing import List, Dict, Union, Optional, Tuple
from argparse import Namespace, ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rdkit.Chem.rdchem import Mol
from matchms import Spectrum
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore

from matplotlib.colors import TABLEAU_COLORS as TABLEAU_COLORS_DICT
from hammer.scalers import SpectraScaler, TransposedSpectraScaler
from hammer.datasets import Dataset
from hammer.dags import LayeredDAG
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


TABLEAU_COLORS: List[str] = list(TABLEAU_COLORS_DICT.keys())


def _visualize_feature(
    samples: Union[List[Mol], List[Spectrum]],
    most_common: Dict[str, np.ndarray],
    top: Dict[str, List[str]],
    feature: Optional[FeatureInterface],
    arguments: Namespace,
) -> None:
    """Visualize a feature."""
    assert isinstance(samples, list)
    assert isinstance(most_common, dict)
    assert isinstance(top, dict)
    assert isinstance(arguments, Namespace)

    normalize_by_parent_mass = False

    for log_intensity, log_mz, norm, normalize_by_parent_mass, include_losses in tqdm(
        [
            (log_intensity, log_mz, norm, normalize_by_parent_mass, include_losses)
            for log_intensity in [True, False]
            for log_mz in [True, False]
            for norm in [True, False]
            for include_losses in [True, False]
        ]
    ):

        if feature is not None:
            name = feature.name()
        else:
            descriptors = []
            if log_intensity:
                descriptors.append("Log Int")

            if log_mz:
                descriptors.append("Log MZ")

            if norm:
                descriptors.append("Normalized")

            if include_losses:
                descriptors.append("With losses")

            if normalize_by_parent_mass:
                descriptors.append("Normalize by parent mass")

            size = 2048

            if include_losses:
                size *= 2

            descriptors.append(str(size))
            name = f"Transposed spectral binning ({', '.join(descriptors)})"

        path = os.path.join(
            arguments.output_directory, f"{name}.{arguments.image_format}"
        )

        if os.path.exists(path):
            continue

        if feature is not None:
            assert isinstance(feature, FeatureInterface)
            assert isinstance(samples[0], Mol)
            x: np.ndarray = feature.transform_molecules(samples)
            if not feature.__class__.is_binary():
                x = RobustScaler().fit_transform(x)

            if np.isnan(x).any():
                return
        else:
            assert isinstance(samples[0], Spectrum)
            # x = SpectraScaler(
            #     bins=2048,
            #     include_losses=include_losses,
            #     normalize=norm,
            #     normalize_by_parent_mass=normalize_by_parent_mass,
            #     log_intensity=log_intensity,
            #     log_mz=log_mz,
            #     verbose=arguments.verbose,
            #     n_jobs=arguments.n_jobs,
            # ).fit_transform(samples).reshape((len(samples), -1))
            x = TransposedSpectraScaler(
                bins=2048,
                include_losses=include_losses,
                normalize=norm,
                normalize_by_parent_mass=normalize_by_parent_mass,
                log_intensity=log_intensity,
                log_mz=log_mz,
                verbose=arguments.verbose,
                n_jobs=arguments.n_jobs,
            ).fit_transform(samples).reshape((len(samples), -1))

            # x = np.hstack(
            #     [
            #         x1.reshape((x1.shape[0], -1)),
            #         x2.reshape((x2.shape[0], -1)),
            #     ]
            # )

        try:
            from MulticoreTSNE import (  # pylint: disable=import-outside-toplevel
                MulticoreTSNE,
            )

            tsne = MulticoreTSNE(
                n_components=2,
                n_jobs=arguments.n_jobs,
                verbose=1,
                n_iter_early_exag=5 if arguments.smoke_test else 250,
                n_iter=10 if arguments.smoke_test else 1000,
            )
        except ImportError:
            from sklearn.manifold import TSNE  # pylint: disable=import-outside-toplevel

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
            x_reduced = x

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

                ax[i, j].set_title(f"{name} - {label_name.capitalize()}")
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
        fig.savefig(path)
        plt.close()


def add_visualize_features_subcommand(visualize_features_parser: ArgumentParser):
    """Add the visualize features sub-command to the parser."""
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


def visualize(arguments: Namespace) -> None:
    """Train the model."""

    dataset: Dataset = build_dataset_from_namespace(namespace=arguments)
    dag: LayeredDAG = dataset.layered_dag()
    samples, labels = dataset.all_samples()

    # We construct the augmentation strategies from the argument parser.
    augmentation_settings: AugmentationSettings = (
        build_augmentation_settings_from_namespace(arguments)
    )

    samples, labels = augmentation_settings.augment(samples, labels)

    # We construct the feature settings from the argument parser.
    feature_settings: FeatureSettings = build_features_settings_from_namespace(
        arguments
    )

    if not feature_settings.includes_features() and not isinstance(
        samples[0], Spectrum
    ):
        feature_settings = FeatureSettings().include_all()

    counter: np.ndarray = dataset.label_counts()

    # Since we can't possibly show all of the superclasses and classes,
    # we will only show the top 'number of colors' of each. Some samples will have multiple
    # pathways, superclasses and classes, so we will only show the most common ones.
    number_of_colors = len(TABLEAU_COLORS)

    # We determine the top 'number_of_colors - 1' most common labels

    top_labels: Dict[str, List[str]] = {}
    for layer_name in tqdm(
        dag.layer_names(),
        desc="Determining the most common labels in layers",
        disable=not arguments.verbose,
        leave=False,
        dynamic_ncols=True,
        unit="layer",
    ):
        counts: List[Tuple[str, int]] = [
            (node_label, counter[dag.node_id(node_label)])
            for node_label in dag.nodes_in_layer(layer_name)
        ]
        sorted_counts = sorted(counts, key=lambda x: x[1], reverse=True)
        clipped_sorted_counts = sorted_counts[: number_of_colors - 1]
        top_labels[layer_name] = [label for label, _ in clipped_sorted_counts]

        if len(clipped_sorted_counts) < number_of_colors - 1:
            top_labels[layer_name].append("Other")

    most_common_labels: Dict[str, List[int]] = {
        layer_name: [] for layer_name in dag.layer_names()
    }
    for one_hot_encoded_y in tqdm(
        labels,
        desc="Determining the most common label per sample",
        disable=not arguments.verbose,
        leave=False,
        dynamic_ncols=True,
        unit="sample",
    ):
        for layer_name in dag.layer_names():
            most_common_label = None
            one_label_in_layer = False
            most_common_count = 0
            for label_name in dag.nodes_in_layer(layer_name):
                label_index = dag.node_id(label_name)
                if one_hot_encoded_y[label_index] == 1:
                    one_label_in_layer = True
                    assert counter[label_index] > 0, (
                        f"We expect the count of label {label_name} to be greater than 0, "
                        f"but it is {counter[label_index]}."
                    )
                if (
                    one_hot_encoded_y[label_index] == 1
                    and counter[label_index] > most_common_count
                ):
                    most_common_label = label_name
                    most_common_count = counter[label_index]

            assert one_label_in_layer, (
                f"We expect at least one label in layer {layer_name} to be 1, "
                f"but none were found."
            )
            assert most_common_label is not None

            try:
                most_common_index = top_labels[layer_name].index(most_common_label)
            except ValueError:
                most_common_index = number_of_colors - 1
            most_common_labels[layer_name].append(most_common_index)

    # We convert the most common labels to numpy arrays
    most_common_labels_array: Dict[str, np.ndarray] = {
        key: np.array(value) for key, value in most_common_labels.items()
    }

    if isinstance(samples[0], Spectrum):
        _visualize_feature(
            samples,
            most_common_labels_array,
            top_labels,
            feature=None,
            arguments=arguments,
        )
    else:
        assert isinstance(samples[0], str)
        canonical_samples: List[str] = into_canonical_smiles(
            samples, n_jobs=arguments.n_jobs, verbose=arguments.verbose
        )
        molecules: List[Mol] = smiles_to_molecules(
            canonical_samples, verbose=arguments.verbose, n_jobs=arguments.n_jobs
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
                most_common_labels_array,
                top_labels,
                feature_class(
                    verbose=arguments.verbose,
                    n_jobs=arguments.n_jobs,
                ),
                arguments,
            )
