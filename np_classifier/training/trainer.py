"""Submodule handling the training of the multi-modal multi-class classifier."""

from np_classifier.training.smiles_dataset import SmilesDataset
from np_classifier.training.model import Classifier


class Trainer:
    """Class handling the training of the multi-modal multi-class classifier."""

    def __init__(self, smiles_dataset: SmilesDataset):
        """Initialize the trainer."""
        assert isinstance(smiles_dataset, SmilesDataset)
        self._smiles_dataset = smiles_dataset

    def train(self):
        """Train the classifier."""
        for train, valid in self._smiles_dataset.train_split():
            classifier = Classifier()
            classifier.train(train, valid)
            classifier.evaluate(train)
            classifier.evaluate(valid)
