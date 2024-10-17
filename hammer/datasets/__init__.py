"""Submodule providing Dataset classes for the Hammer model."""

from typing import List, Type
from hammer.datasets.smiles_dataset import Dataset
from hammer.datasets.npc import NPCDataset, NPCScrapedDataset

AVAILABLE_DATASETS: List[Type[Dataset]] = [NPCDataset, NPCScrapedDataset]

__all__ = ["Dataset", "NPCDataset", "NPCScrapedDataset", "AVAILABLE_DATASETS"]
