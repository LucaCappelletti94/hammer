"""Submodule providing exceptions used in the context of the DAGs."""

from typing import List


class UnknownDAGLayer(ValueError):
    """Exception raised when a DAG layer is not found."""

    def __init__(self, layer_name: str, available_layers: List[str]):
        """Initialize the UnknownDAGLayer exception."""
        super().__init__(
            f"Layer '{layer_name}' not found in the available layers: {available_layers}"
        )


class UnknownDAGNode(ValueError):
    """Exception raised when a DAG node is not found."""

    def __init__(self, node_name: str, layer_name: str, available_nodes: List[str]):
        """Initialize the UnknownDAGNode exception."""
        super().__init__(
            f"Node '{node_name}' not found in layer '{layer_name}' with available nodes: {available_nodes}"
        )


class IllegalLink(ValueError):
    """Exception raised when a link is illegal."""

    def __init__(self, parent: str, child: str):
        """Initialize the IllegalLink exception."""
        super().__init__(
            f"Link from '{parent}' to '{child}' is illegal in the current DAG. "
            "This means that either the DAG is wrong, or the dataset is wrong."
        )
