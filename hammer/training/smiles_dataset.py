"""Submodule providing a class handling the SMILES dataset for training."""

from typing import List, Iterator, Tuple, Dict, Optional, Any, Sequence
import os
from collections import Counter
import compress_json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class Dataset:
    """Class handling the dataset for training."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        number_of_splits: int = 10,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        maximal_number_of_molecules: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the SMILES dataset."""
        local_path = os.path.dirname(os.path.abspath(__file__))
        categorical_smiles: pd.DataFrame = pd.read_csv(
            os.path.join(local_path, "categorical.csv.gz")
        )
        multi_label_smiles: List[Dict[str, Any]] = compress_json.local_load(
            "multi_label.json"
        ) + compress_json.local_load("relabelled.json")

        if maximal_number_of_molecules is not None:
            assert maximal_number_of_molecules < categorical_smiles.shape[0]
            categorical_smiles = categorical_smiles.iloc[:maximal_number_of_molecules]
            multi_label_smiles = []

        # We determine the class, pathway and superclass labels.
        pathway_counts = Counter()
        superclass_counts = Counter()
        class_counts = Counter()

        # First, we add the counts from the categorical smiles.
        pathway_counts.update((pathway for pathway in categorical_smiles.pathway_label))
        superclass_counts.update(
            (superclass for superclass in categorical_smiles.superclass_label)
        )
        class_counts.update((class_ for class_ in categorical_smiles.class_label))

        # Then, we add the counts from the multi-label smiles.
        pathway_counts.update(
            (
                pathway
                for multi_label in multi_label_smiles
                for pathway in multi_label["pathway_labels"]
            )
        )
        superclass_counts.update(
            (
                superclass
                for multi_label in multi_label_smiles
                for superclass in multi_label["superclass_labels"]
            )
        )
        class_counts.update(
            (
                class_
                for multi_label in multi_label_smiles
                for class_ in multi_label["class_labels"]
            )
        )

        # Now that we have accounted for all pathways, superclasses and classes,
        # we filter out the labels that do not appear at least 'number_of_splits + 1' times,
        # as such rare labels cannot be reasonably stratified and we could not evaluate
        # the performance of the model on them, nor reasonably expect for the model to
        # learn to predict them.
        pathways = [
            pathway
            for pathway, count in pathway_counts.items()
            if count >= number_of_splits + 1
        ]
        assert (
            len(pathways) > 1
        ), "No pathway labels appear at least 'number_of_splits + 1' times."
        superclasses = [
            superclass
            for superclass, count in superclass_counts.items()
            if count >= number_of_splits + 1
        ]
        assert (
            len(superclasses) > 1
        ), "No superclass labels appear at least 'number_of_splits + 1' times."
        classes = [
            class_
            for class_, count in class_counts.items()
            if count >= number_of_splits + 1
        ]
        assert (
            len(classes) > 1
        ), "No class labels appear at least 'number_of_splits + 1' times."

        # We sort the labels so that we can execute binary searches on them and
        # to facilitate some other operations.
        self._pathway_names = sorted(pathways)
        self._superclass_names = sorted(superclasses)
        self._class_names = sorted(classes)

        # We filter out the molecules that do not have any of the labels which appear
        # at least 'number_of_splits + 1' times.
        categorical_molecules: pd.DataFrame = pd.DataFrame(
            [
                row
                for _, row in categorical_smiles.iterrows()
                if row.pathway_label in pathways
                and row.superclass_label in superclasses
                and row.class_label in classes
            ]
        )
        multi_label_molecules: List[Dict[str, Any]] = [
            {
                "smiles": multi_label["smiles"],
                "pathway_labels": [
                    pathway
                    for pathway in multi_label["pathway_labels"]
                    if pathway in pathways
                ],
                "superclass_labels": [
                    superclass
                    for superclass in multi_label["superclass_labels"]
                    if superclass in superclasses
                ],
                "class_labels": [
                    class_
                    for class_ in multi_label["class_labels"]
                    if class_ in classes
                ],
            }
            for multi_label in multi_label_smiles
            if any(
                pathway_label in pathways
                for pathway_label in multi_label["pathway_labels"]
            )
            and any(
                superclass_label in superclasses
                for superclass_label in multi_label["superclass_labels"]
            )
            and any(
                class_label in classes for class_label in multi_label["class_labels"]
            )
        ]

        # We determine the number of molecules we have available after filtering.
        number_of_molecules = len(categorical_molecules) + len(multi_label_molecules)
        assert number_of_molecules > 0, "No molecules are available after filtering."

        # We determine the least common class for each molecule.
        least_common_class_label_ids: List[int] = [
            self._class_names.index(categorical_molecule.class_label)
            for categorical_molecule in categorical_molecules.itertuples()
        ] + [
            self._class_names.index(
                min(
                    multi_label_molecule["class_labels"],
                    key=lambda class_label: class_counts[class_label],
                )
            )
            for multi_label_molecule in multi_label_molecules
        ]

        # While it would be quite hard to stratify the dataset based on the combination
        # of pathway, superclass and class labels, we can achieve a reasonable stratification
        # by stratifying based on the least common label, which is the class label which
        # appears the least number of times in the dataset.
        train_indices, test_indices = train_test_split(
            np.arange(number_of_molecules),
            stratify=least_common_class_label_ids,
            test_size=test_size,
            random_state=random_state,
        )

        self._train_indices: np.ndarray = train_indices
        self._test_indices: np.ndarray = test_indices

        # Next, we store the smiles.
        self._smiles: List[str] = list(categorical_molecules.smiles) + [
            multi_label_molecule["smiles"]
            for multi_label_molecule in multi_label_molecules
        ]

        # We convert the smiles to molecules, sanitize them and then convert them back to SMILES.
        assert len(self._smiles) == len(set(self._smiles)), "Duplicate SMILES found."

        # Next, we store the ragged lists of the pathway, superclass and class indices.
        self._pathway_indices: List[List[int]] = [
            [self._pathway_names.index(categorical_molecule.pathway_label)]
            for categorical_molecule in categorical_molecules.itertuples()
        ] + [
            [
                self._pathway_names.index(pathway_label)
                for pathway_label in multi_label_molecule["pathway_labels"]
            ]
            for multi_label_molecule in multi_label_molecules
        ]
        self._superclass_indices: List[List[int]] = [
            [self._superclass_names.index(categorical_molecule.superclass_label)]
            for categorical_molecule in categorical_molecules.itertuples()
        ] + [
            [
                self._superclass_names.index(superclass_label)
                for superclass_label in multi_label_molecule["superclass_labels"]
            ]
            for multi_label_molecule in multi_label_molecules
        ]
        self._class_indices: List[List[int]] = [
            [self._class_names.index(categorical_molecule.class_label)]
            for categorical_molecule in categorical_molecules.itertuples()
        ] + [
            [
                self._class_names.index(class_label)
                for class_label in multi_label_molecule["class_labels"]
            ]
            for multi_label_molecule in multi_label_molecules
        ]

        self._validation_splitter = StratifiedShuffleSplit(
            n_splits=number_of_splits,
            test_size=validation_size,
            random_state=random_state // 2 + 1,
        )
        self._verbose = verbose

    def iter_label_triples(self) -> Iterator[Tuple[str, str, str]]:
        """Iterate over the triples (class, superclass, pathway) in the dataset."""
        for (
            pathway_indices,
            superclass_indices,
            class_indices,
        ) in zip(self._pathway_indices, self._superclass_indices, self._class_indices):
            for pathway_index in pathway_indices:
                for superclass_index in superclass_indices:
                    for class_index in class_indices:
                        yield (
                            self._class_names[class_index],
                            self._superclass_names[superclass_index],
                            self._pathway_names[pathway_index],
                        )

    @property
    def pathway_names(self) -> List[str]:
        """Return the pathway names."""
        return self._pathway_names

    @property
    def superclass_names(self) -> List[str]:
        """Return the superclass names."""
        return self._superclass_names

    @property
    def class_names(self) -> List[str]:
        """Return the class names."""
        return self._class_names

    def _as_numpy_labels(
        self,
        smiles_label_indices: Sequence[int],
    ) -> Dict[str, np.ndarray]:
        """Return the labels as numpy arrays.

        Parameters
        ----------
        smiles_label_indices : List[int]
            The indices of the labels of the molecules.
            While in the case of test or validation smiles these
            indices will be unique, in the case of training smiles
            when the smiles are augmented, the indices will be repeated.
        """
        # Now we allocate and fill the labels.
        labels = {
            "pathway": np.zeros(
                (len(smiles_label_indices), len(self._pathway_names)), dtype=np.uint8
            ),
            "superclass": np.zeros(
                (len(smiles_label_indices), len(self._superclass_names)), dtype=np.uint8
            ),
            "class": np.zeros(
                (len(smiles_label_indices), len(self._class_names)), dtype=np.uint8
            ),
        }

        for i, label_index in enumerate(smiles_label_indices):
            for pathway_index in self._pathway_indices[label_index]:
                labels["pathway"][i, pathway_index] = 1
            for superclass_index in self._superclass_indices[label_index]:
                labels["superclass"][i, superclass_index] = 1
            for class_index in self._class_indices[label_index]:
                labels["class"][i, class_index] = 1

        return labels

    def all_smiles(self) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Return all the smiles and labels."""
        all_labels: Dict[str, np.ndarray] = self._as_numpy_labels(
            np.arange(len(self._smiles))
        )
        return (
            self._smiles,
            all_labels,
        )

    def primary_split(
        self,
    ) -> Tuple[
        Tuple[List[str], Dict[str, np.ndarray]], Tuple[List[str], Dict[str, np.ndarray]]
    ]:
        """Split the dataset into training and test sets."""
        training_molecules: List[str] = [self._smiles[i] for i in self._train_indices]
        test_molecules: List[str] = [self._smiles[i] for i in self._test_indices]
        training_labels: Dict[str, np.ndarray] = self._as_numpy_labels(
            self._train_indices
        )
        test_labels: Dict[str, np.ndarray] = self._as_numpy_labels(self._test_indices)
        return (
            (training_molecules, training_labels),
            (test_molecules, test_labels),
        )

    def train_split(
        self,
    ) -> Iterator[
        Tuple[
            Tuple[List[str], Dict[str, np.ndarray]],
            Tuple[List[str], Dict[str, np.ndarray]],
        ]
    ]:
        """Split the dataset into training and test sets."""
        class_counts: Counter = Counter(
            [
                class_label
                for index in self._train_indices
                for class_label in self._class_indices[index]
            ]
        )

        least_common_class_labels: List[int] = [
            min(
                self._class_indices[index],
                key=lambda class_index: class_counts[class_index],
            )
            for index in self._train_indices
        ]

        # All class counts must be greater or equal to 2, one for the training
        # and one for the validation set.
        for class_label, count in class_counts.items():
            assert (
                count >= 2
            ), f"The class label '{class_label}' has only {count} molecules."

        for train_indices, validation_indices in tqdm(
            self._validation_splitter.split(
                np.arange(len(self._train_indices)), least_common_class_labels
            ),
            total=self._validation_splitter.get_n_splits(),
            desc="Training and validation splits",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="split",
        ):
            training_molecules: List[str] = [
                self._smiles[i] for i in self._train_indices[train_indices]
            ]
            validation_molecules: List[str] = [
                self._smiles[i] for i in self._train_indices[validation_indices]
            ]
            training_labels: Dict[str, np.ndarray] = self._as_numpy_labels(
                self._train_indices[train_indices]
            )
            validation_labels: Dict[str, np.ndarray] = self._as_numpy_labels(
                self._train_indices[validation_indices]
            )
            yield (
                (training_molecules, training_labels),
                (validation_molecules, validation_labels),
            )
