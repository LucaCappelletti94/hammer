"""Submodule providing a graph interface."""

from abc import abstractmethod
from typing import List, Tuple
from dict_hash import Hashable, sha256
import numpy as np
from scipy.sparse import csr_matrix, eye, diags, coo_matrix


class Graph(Hashable):
    """Abstract class defining the Graph interface."""

    @abstractmethod
    def nodes(self) -> List[str]:
        """Return the names of the nodes in the graph."""

    def node_id(self, node_name: str) -> int:
        """Return the id of a node."""
        return self.nodes().index(node_name)

    def number_of_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes())

    def adjacency_matrix(self) -> csr_matrix:
        """Return the adjacency matrix of the graph."""
        nodes: List[str] = self.nodes()
        cooccurrences: List[Tuple[int, int]] = []
        for i, node in enumerate(nodes):
            for neighbor in self.outbounds(node):
                j = nodes.index(neighbor)
                cooccurrences.append((i, j))

        adjacency_matrix: csr_matrix = coo_matrix(
            (
                np.ones(len(cooccurrences), dtype=np.float32),
                ([x[0] for x in cooccurrences], [x[1] for x in cooccurrences]),
            ),
            shape=(len(nodes), len(nodes)),
        ).tocsr()

        return adjacency_matrix

    def get_node_out_degree(self, node_name: str) -> int:
        """Return the out degree of a node."""
        return len(self.outbounds(node_name))

    def get_node_in_degree(self, node_name: str) -> int:
        """Return the in degree of a node."""
        occurrences = 0
        for node in self.nodes():
            if node_name in self.outbounds(node):
                occurrences += 1
        return occurrences

    @abstractmethod
    def outbounds(self, node_name: str) -> List[str]:
        """Return the nodes that a node points to."""

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a hash that is consistent across runs."""
        return sha256(
            {
                "nodes": self.nodes,
                "adjacency_matrix": self.adjacency_matrix,
            },
            use_approximation=use_approximation,
        )

    def symmetric_laplacian(self) -> csr_matrix:
        """Return the symmetric Laplacian of the graph."""
        adjacency = self.adjacency_matrix()
        adjacency_plus_identity = adjacency + eye(adjacency.shape[0])
        d = diags(
            np.power(np.array(adjacency_plus_identity.sum(axis=1)), -0.5).flatten(), 0
        )
        return adjacency_plus_identity.dot(d).transpose().dot(d).tocsr().astype(np.float32)

    def laplacian(self) -> csr_matrix:
        """Return the Laplacian of the graph."""
        adjacency = self.adjacency_matrix()
        adjacency_plus_identity = adjacency + eye(adjacency.shape[0])
        d = diags(np.power(np.array(adjacency_plus_identity.sum(1)), -1).flatten(), 0)
        return d.dot(adjacency_plus_identity).tocsr().astype(np.float32)

    def transposed_laplacian(self) -> csr_matrix:
        """Return the transposed Laplacian of the graph."""
        adjacency_transposed = self.adjacency_matrix().T
        adjacency_transposed_plus_identity = adjacency_transposed + eye(
            adjacency_transposed.shape[0]
        )
        d = diags(
            np.power(np.array(adjacency_transposed_plus_identity.sum(1)), -1).flatten(),
            0,
        )
        return d.dot(adjacency_transposed_plus_identity).tocsr().astype(np.float32)
