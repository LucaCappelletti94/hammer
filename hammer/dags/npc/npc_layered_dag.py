"""Submodule providing the Layered DAG defined by the original NP Classifier (adjusted)."""

from typing import List, Dict
import compress_json
from hammer.dags.layered_dag import LayeredDAG
from hammer.exceptions import UnknownDAGLayer, UnknownDAGNode


class NPCDAG(LayeredDAG):
    """Class defining the Layered DAG for the NP Classifier (adjusted)."""

    def __init__(self) -> None:
        """Initialize the Layered DAG for the NP Classifier (adjusted)."""
        self._dag: Dict[str, Dict[str, List[str]]] = compress_json.local_load(  # type: ignore
            "npc_layered_dag.json"
        )
        self._pathway_names = compress_json.local_load("npc_pathway_names.json")
        self._superclass_names = compress_json.local_load("npc_superclass_names.json")
        self._class_names = compress_json.local_load("npc_class_names.json")
        self._nodes: List[str] = (
            self._pathway_names + self._superclass_names + self._class_names
        )

    def nodes(self) -> List[str]:
        """Return the nodes in the DAG."""
        return self._nodes

    def layer_names(self) -> List[str]:
        """Return the names of the layers in the DAG."""
        return [
            "pathways",
            "superclasses",
            "classes",
        ]

    def nodes_in_layer(self, layer_name: str) -> List[str]:
        """Return the names of the nodes in a layer."""
        if layer_name == "pathways":
            return self._pathway_names
        if layer_name == "superclasses":
            return self._superclass_names
        if layer_name == "classes":
            return self._class_names

        raise UnknownDAGLayer(layer_name, self.layer_names())

    def outbounds(self, node_name: str) -> List[str]:
        """Return the parents of a node in a layer."""
        for layer_name in self.layer_names():
            if node_name in self.nodes_in_layer(layer_name):
                if layer_name == "pathways":
                    return []
                return self._dag[layer_name][node_name]
        raise UnknownDAGNode(node_name, self.nodes())
