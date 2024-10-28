"""DAG whose nodes are primarily organized into layers."""

from abc import abstractmethod
from typing import List
from hammer.dags.dag import DAG


class LayeredDAG(DAG):
    """Class defining a DAG primarily organized into layers."""

    def nodes(self) -> List[str]:
        """Return the nodes in the DAG."""
        return [
            node for layer in self.layer_names() for node in self.nodes_in_layer(layer)
        ]

    @abstractmethod
    def layer_names(self) -> List[str]:
        """Return the names of the layers in the DAG."""

    @abstractmethod
    def nodes_in_layer(self, layer_name: str) -> List[str]:
        """Return the nodes in a layer."""
