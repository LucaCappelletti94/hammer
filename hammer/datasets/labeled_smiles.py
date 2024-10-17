"""Submodule defining a class for storing a labeled SMILES."""

from typing import Dict, List, Type, Optional
from collections import Counter
from rdkit.Chem import MolFromSmiles  # pylint: disable=import-error, no-name-in-module
from dict_hash import Hashable, sha256
import numpy as np
from hammer.layered_dags import LayeredDAG
from hammer.exceptions import IllegalLink


class LabeledSMILES(Hashable):
    """Class for storing a labeled SMILES."""

    def __init__(self, smiles: str, labels: Dict[str, List[str]]):
        """Initialize the LabeledSMILES object."""
        # Some defensive programming to ensure that the input is valid.
        assert isinstance(smiles, str), "SMILES must be a string."
        assert MolFromSmiles(smiles) is not None, "Invalid SMILES."
        assert isinstance(labels, dict), "Labels must be a dictionary."
        assert all(
            isinstance(k, str) for k in labels.keys()
        ), "Keys of labels must be strings."
        assert all(
            isinstance(v, list) for v in labels.values()
        ), "Values of labels must be lists."
        assert all(
            all(isinstance(s, str) for s in v) for v in labels.values()
        ), "Values of labels must be lists of strings."
        self._smiles: str = smiles
        self._labels: Dict[str, List[str]] = labels

    @property
    def smiles(self) -> str:
        """Return the SMILES string."""
        return self._smiles

    @property
    def labels(self) -> Dict[str, List[str]]:
        """Return the labels."""
        return self._labels

    # pylint: disable=protected-access
    def merge_labels(self, other: "LabeledSMILES") -> "LabeledSMILES":
        """Merge the labels of the current labeled SMILES with another one."""
        if self.smiles != other.smiles:
            raise ValueError("Cannot merge labels of different SMILES.")
        new_labels = {
            layer_name: list(
                set(
                    self._labels.get(layer_name, []) + other._labels.get(layer_name, [])
                )
            )
            for layer_name in set(self._labels.keys()) | set(other._labels.keys())
        }
        return LabeledSMILES(smiles=self.smiles, labels=new_labels)

    def least_common_label(self, counters: Counter, leaf_layer_name: str) -> str:
        """Returns the least common label in the leaf layer."""
        return min(self._labels[leaf_layer_name], key=lambda label: counters[label])

    def has_missing_labels(self, layer_names: List[str]) -> bool:
        """Returns whether there are missing labels for the given layers."""
        return any(
            layer_name not in self._labels or len(self._labels[layer_name]) == 0
            for layer_name in layer_names
        )

    # pylint: disable=protected-access
    def first_discordant_layer(
        self, other: "LabeledSMILES", layer_names: List[str]
    ) -> Optional[str]:
        """Returns the first discordant layer between the two labeled SMILES."""
        for layer_name in layer_names:
            if (layer_name not in self._labels and layer_name in other._labels) or (
                layer_name in self._labels and layer_name not in other._labels
            ):
                return layer_name
            if any(
                label not in other._labels[layer_name]
                for label in self._labels.get(layer_name, [])
            ) and any(
                label not in self._labels[layer_name]
                for label in other._labels.get(layer_name, [])
            ):
                return layer_name
        return None

    def iter_paths(self, layer_names: List[str]) -> List[List[str]]:
        """Returns over all paths defined by the current SMILES labels."""
        # All paths start from the last layer, i.e. the leaf, and
        # end at the roots. Since these paths represent a DAG, it
        # is expected that the paths are acyclic, and that there
        # will be multiple paths to the root.
        paths = []
        for layer_name in reversed(layer_names):
            if len(paths) == 0:
                paths = [[node] for node in self._labels[layer_name]]
                continue
            new_paths = []
            for node in self._labels[layer_name]:
                for path in paths:
                    new_paths.append(path + [node])
            paths = new_paths
        return paths

    def numpy_labels(self, layered_dag: Type[LayeredDAG]) -> Dict[str, np.ndarray]:
        """Return the labels as numpy arrays."""
        labels: Dict[str, np.ndarray] = {
            layer_name: np.zeros(layered_dag.get_layer_size(layer_name), dtype=np.uint8)
            for layer_name in layered_dag.get_layer_names()
        }

        previous_layer_name: Optional[str] = None
        for layer_name, nodes in self._labels.items():
            layer: List[str] = layered_dag.get_layer(layer_name)
            for node in nodes:
                if previous_layer_name is not None:
                    if not any(
                        layered_dag.has_edge(
                            src_node_name=src,
                            dest_node_name=node,
                            layer_name=previous_layer_name,
                        )
                        for src in self._labels[previous_layer_name]
                    ):
                        composide_src = "or ".join(self._labels[previous_layer_name])
                        raise IllegalLink(composide_src, node)

                labels[layer_name][layer.index(node)] = 1

        return labels

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a hash that is consistent across different Python versions."""
        return sha256(
            {
                "smiles": self._smiles,
                "labels": self._labels,
            },
            use_approximation=use_approximation,
        )

    def __str__(self):
        return f"SMILES: {self._smiles}, Labels: {self._labels}"
