"""Submodule with DAG interface and its concrete implementations."""

from hammer.layered_dags.layered_dag import LayeredDAG
from hammer.layered_dags.npc import NPCLayeredDAG

__all__ = ["LayeredDAG", "NPCLayeredDAG"]
