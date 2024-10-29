"""Submodule providing a Layered DAG interface for the Hammer model."""

from typing import Iterable
from hammer.dags.graph import Graph


class DAG(Graph):
    """Abstract class defining the Layered DAG interface."""

    def number_of_paths_from_node(self, node_name: str) -> int:
        """Return the number of paths that exist from a node in a layer."""
        if self.is_sink_node(node_name):
            return 1

        return sum(
            self.number_of_paths_from_node(parent)
            for parent in self.outbounds(node_name)
        )

    def is_sink_node(self, node_name: str) -> bool:
        """Return whether a node is a sink node."""
        return (
            self.get_node_out_degree(node_name) == 0
            and self.get_node_in_degree(node_name) > 0
        )

    def is_source_node(self, node_name: str) -> bool:
        """Return whether a node is a source node."""
        return (
            self.get_node_in_degree(node_name) == 0
            and self.get_node_out_degree(node_name) > 0
        )

    def iter_sink_nodes(self) -> Iterable[str]:
        """Return an iterator over the sink nodes."""
        for node in self.nodes():
            if self.is_sink_node(node):
                yield node

    def iter_source_nodes(self) -> Iterable[str]:
        """Return an iterator over the source nodes."""
        for node in self.nodes():
            if self.is_source_node(node):
                yield node

    @property
    def number_of_paths(self) -> int:
        """Return the number of paths from leafs to root that exist in the DAG."""
        return sum(
            self.number_of_paths_from_node(node_name)
            for node_name in self.iter_source_nodes()
        )
