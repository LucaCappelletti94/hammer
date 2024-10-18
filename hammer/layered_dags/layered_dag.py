"""Submodule providing a Layered DAG interface for the Hammer model."""

from typing import List, Dict
from abc import abstractmethod
from dict_hash import Hashable, sha256
import numpy as np


class LayeredDAG(Hashable):
    """Abstract class defining the Layered DAG interface."""

    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Return the names of the layers in the DAG."""

    @abstractmethod
    def get_layer(self, layer_name: str) -> List[str]:
        """Return the names of the nodes in a layer."""

    @property
    def leaf_layer_name(self) -> str:
        """Return the name of the leaf layer."""
        return self.get_layer_names()[-1]

    @property
    def root_layer_name(self) -> str:
        """Return the name of the root layer."""
        return self.get_layer_names()[0]

    def iter_root_nodes(self) -> List[str]:
        """Return an iterator over the root nodes."""
        return self.get_layer(self.root_layer_name)

    def has_node(self, node_name: str, layer_name: str) -> bool:
        """Return whether a node is in a layer."""
        return node_name in self.get_layer(layer_name)

    def get_layer_size(self, layer_name: str) -> int:
        """Return the number of nodes in a layer."""
        return len(self.get_layer(layer_name))

    def get_parent_layer_name(self, layer_name: str) -> str:
        """Return the name of the parent layer for a layer."""
        layer_names = self.get_layer_names()
        layer_index = layer_names.index(layer_name)
        if layer_index == 0:
            raise ValueError(f"The first layer {layer_name} does not have a parent.")
        return layer_names[layer_index - 1]

    @abstractmethod
    def get_parents(self, node_name: str, layer_name: str) -> List[str]:
        """Return the parents of a node in a layer."""

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a hash that is consistent across different Python versions."""
        return sha256(
            {
                "layer_names": self.get_layer_names(),
                "layers": {
                    layer_name: self.get_layer(layer_name)
                    for layer_name in self.get_layer_names()
                },
            },
            use_approximation=use_approximation,
        )

    def layer_adjacency_matrix(self, layer_name: str) -> np.ndarray:
        """Return the adjacency matrix for a layer to the parent layer."""
        layer = self.get_layer(layer_name)
        parent_layer = self.get_parent_layer_name(layer_name)
        parent_layer_size = self.get_layer_size(parent_layer)
        adjacency_matrix = np.zeros((len(layer), parent_layer_size), dtype=np.uint8)

        for i, node_name in enumerate(layer):
            parents = self.get_parents(node_name, layer_name)
            for parent in parents:
                j = self.get_layer(parent_layer).index(parent)
                adjacency_matrix[i, j] = 1

        return adjacency_matrix

    def layer_adjacency_matrices(self) -> Dict[str, np.ndarray]:
        """Return the adjacency matrices for all layers that have a parent."""
        return {
            layer_name: self.layer_adjacency_matrix(layer_name)
            for layer_name in self.get_layer_names()[1:]
        }

    def number_of_paths_from_node(self, node_name: str, layer_name: str) -> int:
        """Return the number of paths that exist from a node in a layer."""
        if layer_name == self.get_layer_names()[0]:
            return 1

        parents = self.get_parents(node_name, layer_name)
        return sum(
            self.number_of_paths_from_node(
                parent, self.get_parent_layer_name(layer_name)
            )
            for parent in parents
        )

    def has_edge(self, src_node_name: str, dst_node_name: str, layer_name: str) -> bool:
        """Return whether an edge exists between two nodes in the DAG."""
        return dst_node_name in self.get_parents(src_node_name, layer_name)

    @property
    def number_of_paths(self) -> int:
        """Return the number of paths from leafs to root that exist in the DAG."""
        return sum(
            self.number_of_paths_from_node(node_name, self.leaf_layer_name)
            for node_name in self.get_layer(self.leaf_layer_name)
        )
