"""Submodule defining the DAG interface and concrete implementations of Layered DAGS for the Hammer model."""

from hammer.training.layered_dags.layered_dag import LayeredDAG
from hammer.training.layered_dags.npc import NPCLayeredDAG

__all__ = ["LayeredDAG", "NPCLayeredDAG"]
