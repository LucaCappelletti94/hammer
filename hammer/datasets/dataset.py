"""Submodule providing an abstract class handling the datasets for training."""

from typing import List, Iterator, Tuple, Optional, Any
from abc import abstractmethod
import numpy as np
from dict_hash import Hashable, sha256
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split  # type: ignore
from hammer.dags import LayeredDAG


class Dataset(Hashable):
    """Class handling the dataset for training."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        maximal_number_of_molecules: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the dataset."""
        self._maximal_number_of_molecules = maximal_number_of_molecules
        self._random_state = random_state
        self._verbose = verbose

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Return the name of the dataset."""

    @staticmethod
    @abstractmethod
    def description() -> str:
        """Return a description of the dataset."""

    @abstractmethod
    def layered_dag(self) -> LayeredDAG:
        """Return the layered DAG for the dataset."""

    @abstractmethod
    def iter_samples(self) -> Iterator[Tuple[Any, np.ndarray]]:
        """Iterate over the one-hot encoded labels in the dataset."""

    @abstractmethod
    def number_of_samples(self) -> int:
        """Return the number of samples in the dataset."""

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a consistent hash for the dataset."""
        return sha256(
            {
                "name": self.name(),
            },
            use_approximation=use_approximation,
        )

    def label_counts(self) -> np.ndarray:
        """Return the counts of the labels in the dataset."""
        total: np.ndarray = np.zeros(self.layered_dag().number_of_nodes())
        for _, labels in self.iter_samples():
            total += labels
        return total

    def all_samples(self) -> Tuple[List[Any], np.ndarray]:
        """Return all the smiles and labels."""
        all_labels: List[np.ndarray] = []
        samples: List[Any] = []
        for sample, labels in self.iter_samples():
            samples.append(sample)
            all_labels.append(labels)

        return samples, np.vstack(all_labels)

    def primary_split(
        self, test_size: float
    ) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
        """Split the dataset into training and test sets."""
        counters: np.ndarray = self.label_counts()

        # We verify that all terms appear at least two times in the dataset,
        # or we cannot split the dataset into training and test sets in a stratified manner.
        rare_terms: List[Tuple[str, int]] = []
        for count, node_name in zip(counters, self.layered_dag().nodes()):
            if count > 0 and count < 2:
                rare_terms.append((node_name, count))

        if rare_terms:
            raise ValueError(
                f"Terms appear less than two times in the dataset: {rare_terms}"
            )

        # We identify the least common labels.
        least_common_labels: np.ndarray = np.zeros(
            self.number_of_samples(), dtype=np.uint8
        )
        for i, (_, labels) in enumerate(self.iter_samples()):
            least_common_labels[i] = min(
                np.argwhere(labels == 1)[:, 0],
                key=lambda i: counters[i],
            )

        # Having now identified the least common labels in the leaf layer, we
        # can use these to split the dataset into training and test sets in a
        # stratified manner.

        train_indices, test_indices = train_test_split(
            np.arange(self.number_of_samples()),
            stratify=least_common_labels,
            test_size=test_size,
            random_state=self._random_state,
        )

        smiles, labels = self.all_samples()

        return (
            (
                [smiles[i] for i in train_indices],
                np.vstack([labels[i] for i in train_indices]),
            ),
            (
                [smiles[i] for i in test_indices],
                np.vstack([labels[i] for i in test_indices]),
            ),
        )

    def train_split(
        self, number_of_holdouts: int, validation_size: float, test_size: float
    ) -> Iterator[
        Tuple[
            Tuple[List[Any], np.ndarray],
            Tuple[List[Any], np.ndarray],
        ]
    ]:
        """Split the dataset into training and test sets."""
        (train_smiles, train_labels), (_test_smiles, _test_labels) = self.primary_split(
            test_size=test_size,
        )
        counters: np.ndarray = np.sum(train_labels, axis=0)

        # We verify that all terms appear at least two times in the dataset,
        # or we cannot split the dataset into training and test sets in a stratified manner.
        rare_terms: List[Tuple[str, int]] = []
        for count, node_name in zip(counters, self.layered_dag().nodes()):
            if count > 0 and count < 2:
                rare_terms.append((node_name, count))

        if rare_terms:
            raise ValueError(
                f"Terms appear less than two times in the dataset: {rare_terms}"
            )

        # We identify the least common labels.
        least_common_labels: np.ndarray = np.zeros(len(train_smiles), dtype=np.uint8)
        for i, train_label in enumerate(train_labels):
            least_common_labels[i] = min(
                np.argwhere(train_label == 1)[:, 0],
                key=lambda i: counters[i],
            )

        splitter: StratifiedShuffleSplit = StratifiedShuffleSplit(
            n_splits=number_of_holdouts,
            test_size=validation_size,
            random_state=self._random_state,
        )

        for train_indices, validation_indices in tqdm(
            splitter.split(train_smiles, least_common_labels),
            total=number_of_holdouts,
            desc="Holdouts",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="holdout",
        ):
            yield (
                (
                    [train_smiles[i] for i in train_indices],
                    np.vstack([train_labels[i] for i in train_indices]),
                ),
                (
                    [train_smiles[i] for i in validation_indices],
                    np.vstack([train_labels[i] for i in validation_indices]),
                ),
            )
