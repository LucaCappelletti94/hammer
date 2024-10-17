"""Submodule providing the NP Classifier dataset."""

from hammer.datasets.npc.npc_dataset import NPCDataset
from hammer.datasets.npc.npc_scraped_dataset import NPCScrapedDataset

__all__ = ["NPCDataset", "NPCScrapedDataset"]
