"""Submodule providing Dataset classes for the Hammer model."""

from typing import List, Type
from hammer.training.datasets.smiles_dataset import Dataset
from hammer.training.datasets.npc import NPCDataset

AVAILABLE_DATASETS: List[Type[Dataset]] = [NPCDataset]

__all__ = ["Dataset", "NPCDataset"]
