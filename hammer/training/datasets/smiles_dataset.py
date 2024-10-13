"""Submodule providing an abstract class handling the SMILES dataset for training."""

from typing import List, Iterator, Tuple, Dict, Optional, Type, Set
from abc import abstractmethod
from collections import Counter
import numpy as np
from dict_hash import Hashable, sha256
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from hammer.training.layered_dags import LayeredDAG
from hammer.training.datasets.labeled_smiles import LabeledSMILES
from hammer.exceptions import UnknownDAGNode


class Dataset(Hashable):
    """Class handling the dataset for training."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        maximal_number_of_molecules: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the SMILES dataset."""
        self._maximal_number_of_molecules = maximal_number_of_molecules
        self._random_state = random_state
        self._verbose = verbose

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Return the name of the dataset."""

    @abstractmethod
    def layered_dag(self) -> Type[LayeredDAG]:
        """Return the layered DAG for the dataset."""

    @abstractmethod
    def iter_labeled_smiles(self) -> Iterator[LabeledSMILES]:
        """Iterate over the labeled SMILES in the dataset."""

    @abstractmethod
    def number_of_smiles(self) -> int:
        """Return the number of SMILES in the dataset."""

    def label_counters(self) -> Dict[str, Counter]:
        """Return the counters for all the labels in the associated dataset."""
        counters: Dict[str, Counter] = {
            layer_name: Counter() for layer_name in self.layered_dag().get_layer_names()
        }
        for labeled_smiles in self.iter_labeled_smiles():
            for layer_name, labels in labeled_smiles.labels.items():
                for label in labels:
                    if not self.layered_dag().has_node(label, layer_name):
                        raise UnknownDAGNode(
                            node_name=label,
                            layer_name=layer_name,
                            available_nodes=self.layered_dag().get_layer(layer_name),
                        )
                counters[layer_name].update(labels)
        return counters

    def iter_labeled_smiles_with_minimum_count(
        self, minimum_count: int
    ) -> Iterator[LabeledSMILES]:
        """Iterates over labeled SMILES with labels appearing at least 'minimum_count' times."""
        counters: Dict[str, Counter] = self.label_counters()
        for labeled_smiles in self.iter_labeled_smiles():
            if all(
                all(counters[layer_name][label] >= minimum_count for label in labels)
                for layer_name, labels in labeled_smiles.labels.items()
            ):
                yield labeled_smiles

    def dag_coverage(self, minimum_count: int = 0) -> float:
        """Returns the percentage of paths in the DAG that are covered by the dataset.

        Parameters
        ----------
        minimum_count : int = 0
            The minimum number of times a label must appear in the dataset to be considered.
        """
        paths_in_dataset: Set[Tuple[str]] = {
            tuple(path)
            for labeled_smiles in self.iter_labeled_smiles_with_minimum_count(
                minimum_count
            )
            for path in labeled_smiles.iter_paths(self.layered_dag().get_layer_names())
        }

        return len(paths_in_dataset) / self.layered_dag().number_of_paths

    def dag_layer_coverage(self, minimum_count: int = 0) -> Dict[str, float]:
        """Returns nodes coverage percentage in each DAG layer."""
        counters: Dict[str, Counter] = self.label_counters()
        return {
            layer_name: sum(
                count >= minimum_count for count in counters[layer_name].values()
            )
            / self.layered_dag().get_layer_size(layer_name)
            for layer_name in self.layered_dag().get_layer_names()
        }

    def all_smiles(self) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Return all the smiles and labels."""
        number_of_smiles = self.number_of_smiles()
        dag: Type[LayeredDAG] = self.layered_dag()
        all_labels: Dict[str, np.ndarray] = {
            layer_name: np.zeros(
                (number_of_smiles, dag.get_layer_size(layer_name)),
            )
            for layer_name in dag.get_layer_names()
        }
        smiles: List[str] = []
        for i, labeled_smiles in enumerate(self.iter_labeled_smiles()):
            smiles.append(labeled_smiles.smiles)
            for layer_name, labels in labeled_smiles.labels.items():
                layer: List[str] = dag.get_layer(layer_name)
                for label in labels:
                    all_labels[layer_name][i, layer.index(label)] = 1
        return smiles, all_labels

    def primary_split(
        self, test_size: float
    ) -> Tuple[
        Tuple[List[str], Dict[str, np.ndarray]], Tuple[List[str], Dict[str, np.ndarray]]
    ]:
        """Split the dataset into training and test sets."""
        counters: Dict[str, Counter] = self.label_counters()

        # We verify that all terms appear at least two times in the dataset,
        # or we cannot split the dataset into training and test sets in a stratified manner.
        for layer_name, counter in counters.items():
            for node_name, count in counter.items():
                if count < 2:
                    raise ValueError(
                        f"The label '{node_name}' in layer "
                        f"'{layer_name}' appears only {count} times."
                    )

        leaf_layer_name: str = self.layered_dag().leaf_layer_name
        leaf_layer: List[str] = self.layered_dag().get_layer(leaf_layer_name)
        leaf_layer_counter: Counter = counters[leaf_layer_name]
        least_common_leaf_labels: np.ndarray = np.fromiter(
            (
                leaf_layer.index(
                    labelled_smiles.least_common_label(
                        leaf_layer_counter, leaf_layer_name
                    )
                )
                for labelled_smiles in self.iter_labeled_smiles()
            ),
            dtype=int,
        )

        # Having now identified the least common labels in the leaf layer, we
        # can use these to split the dataset into training and test sets in a
        # stratified manner.

        train_indices, test_indices = train_test_split(
            np.arange(self.number_of_smiles()),
            stratify=least_common_leaf_labels,
            test_size=test_size,
            random_state=self._random_state,
        )

        smiles, labels = self.all_smiles()

        training_molecules: List[str] = [smiles[i] for i in train_indices]
        test_molecules: List[str] = [smiles[i] for i in test_indices]
        training_labels: Dict[str, np.ndarray] = {
            layer_name: labels[layer_name][train_indices]
            for layer_name in self.layered_dag().get_layer_names()
        }
        test_labels: Dict[str, np.ndarray] = {
            layer_name: labels[layer_name][test_indices]
            for layer_name in self.layered_dag().get_layer_names()
        }

        return (
            (training_molecules, training_labels),
            (test_molecules, test_labels),
        )

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return consistent hash of the current object.

        Parameters
        ----------
        use_approximation : bool = False
            If True, the hash can be approximated. This is useful when the
            hash is too long and we want to use a shorter version of it.

        Returns
        -------
        A consistent hash of the object.
        """
        return sha256(
            {
                "dag": self.layered_dag(),
                "smiles": list(self.iter_labeled_smiles()),
            },
            use_approximation=use_approximation,
        )

    def train_split(
        self, number_of_holdouts: int, validation_size: float, test_size: float
    ) -> Iterator[
        Tuple[
            Tuple[List[str], Dict[str, np.ndarray]],
            Tuple[List[str], Dict[str, np.ndarray]],
        ]
    ]:
        """Split the dataset into training and test sets."""
        (train_smiles, train_labels), (_test_smiles, _test_labels) = self.primary_split(
            test_size=test_size,
        )
        leaf_layer_train_labels = train_labels[self.layered_dag().leaf_layer_name]
        leaf_counts = np.sum(leaf_layer_train_labels, axis=0)
        least_common_class_labels = np.fromiter(
            (
                min(
                    np.argwhere(leaf_layer_train_labels[training_sample_index])[
                        :, 0
                    ],
                    key=lambda i: leaf_counts[i],
                )
                for training_sample_index in range(len(train_smiles))
            ),
            dtype=int,
        )

        splitter: StratifiedShuffleSplit = StratifiedShuffleSplit(
            n_splits=number_of_holdouts,
            test_size=validation_size,
            random_state=self._random_state,
        )

        for train_indices, validation_indices in tqdm(
            splitter.split(train_smiles, least_common_class_labels),
            total=number_of_holdouts,
            desc="Holdouts",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="holdout",
        ):
            training_molecules: List[str] = [train_smiles[i] for i in train_indices]
            validation_molecules: List[str] = [
                train_smiles[i] for i in validation_indices
            ]
            training_labels: Dict[str, np.ndarray] = {
                layer_name: train_labels[layer_name][train_indices]
                for layer_name in self.layered_dag().get_layer_names()
            }
            validation_labels: Dict[str, np.ndarray] = {
                layer_name: train_labels[layer_name][validation_indices]
                for layer_name in self.layered_dag().get_layer_names()
            }
            yield (
                (training_molecules, training_labels),
                (validation_molecules, validation_labels),
            )
