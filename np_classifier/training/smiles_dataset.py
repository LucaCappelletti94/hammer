"""Submodule providing a class handling the SMILES dataset for training."""

from typing import List, Iterator, Tuple, Dict, Set
import os
from multiprocessing import Pool
import compress_json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from np_classifier.training.smiles import Smiles
from np_classifier.training.augmentation_strategies import (
    generate_demethoxylated_homologues,
    generate_methoxylated_homologues,
)

def _get_fingerprints(smile: (Smiles, int, int)) -> Dict[str, np.ndarray]:
    """Return the fingerprints of a SMILES."""
    smile, radius, n_bits = smile
    return smile.fingerprint(radius=radius, n_bits=n_bits)

class SmilesDataset:
    """Class handling the SMILES dataset for training."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        number_of_splits: int = 10,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        radius: int = 3,
        n_bits: int = 2048,
        verbose: bool = True,
    ):
        """Initialize the SMILES dataset."""
        local_path = os.path.dirname(os.path.abspath(__file__))
        categorical_smiles = os.path.join(local_path, "categorical.csv.gz")
        multi_label_smiles = os.path.join(local_path, "multi_label.json")
        smiles: List[Smiles] = [
            Smiles(
                smiles=row.smiles,
                pathway_labels=[row.pathway_label],
                superclass_labels=[row.superclass_label],
                class_labels=[row.class_label],
            )
            for row in pd.read_csv(categorical_smiles).itertuples()
        ] + [
            Smiles(
                smiles=entry["smiles"],
                pathway_labels=entry["pathway_labels"],
                superclass_labels=entry["superclass_labels"],
                class_labels=entry["class_labels"],
            )
            for entry in compress_json.load(multi_label_smiles)
        ]
        # While it would be quite hard to stratify the dataset based on the combination
        # of pathway, superclass and class labels, we can achieve a reasonable stratification
        # by stratifying based on the first class label.
        train_indices, test_indices = train_test_split(
            np.arange(len(smiles)),
            stratify=[smiles.first_class_label for smiles in smiles],
            test_size=test_size,
            random_state=random_state,
        )

        self._test_smiles: List[Smiles] = [smiles[i] for i in test_indices]

        # We augment the training set.
        smiles_in_training_set: Set[str] = {smiles[i].smiles for i in train_indices}
        training_smiles: List[Smiles] = [smiles[i] for i in train_indices]
        augmented_smiles: List[Smiles] = []

        with Pool() as pool:
            demethoxylated_homologues: List[List[str]] = list(
                tqdm(
                    pool.imap(generate_demethoxylated_homologues, (smile.smiles for smile in training_smiles)),
                    desc="Generating demethoxylated homologues",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not verbose,
                    total=len(training_smiles),
                    unit="molecule",
                )
            )
            methoxylated_homologues: List[List[str]] = list(
                tqdm(
                    pool.imap(generate_methoxylated_homologues, (smile.smiles for smile in training_smiles)),
                    desc="Generating methoxylated homologues",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not verbose,
                    total=len(training_smiles),
                    unit="molecule",
                )
            )

        for i, smile in enumerate(training_smiles):
            for homologue in demethoxylated_homologues[i] + methoxylated_homologues[i]:
                if homologue not in smiles_in_training_set:
                    augmented_smiles.append(smile.into_homologue(homologue))
                    smiles_in_training_set.add(homologue)

        print(f"Augmented training set with {len(augmented_smiles)} homologues.")
        print(f"Original training set size: {len(training_smiles)}")

        self._training_smiles: List[Smiles] = training_smiles + augmented_smiles
        self._validation_splitter = StratifiedShuffleSplit(
            n_splits=number_of_splits,
            test_size=validation_size,
            random_state=random_state // 2,
        )
        self._verbose = verbose
        self._radius = radius
        self._n_bits = n_bits

    def smiles_to_dataset(
        self, smiles: List[Smiles]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Convert a list of SMILES to a dataset."""
        number_of_smiles = len(smiles)
        
        with Pool() as pool:
            fingerprints = list(
                tqdm(
                    pool.imap(_get_fingerprints, ((smile, self._radius, self._n_bits) for smile in smiles)),
                    desc="Generating fingerprints",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not self._verbose,
                    total=number_of_smiles,
                    unit="molecule",
                )
            )

        smile_labels = smiles[0].labels()
        dataset = {
            key: np.zeros((number_of_smiles, value.shape[1]), dtype=np.float32)
            for key, value in fingerprints[0].items()
        }
        labels = {
            key: np.zeros((number_of_smiles, value.shape[0]), dtype=np.float32)
            for key, value in smile_labels.items()
        }

        for i, (smile, fingerprint) in enumerate(zip(smiles, fingerprints)):
            label = smile.labels()
            for key in dataset:
                dataset[key][i] = fingerprint[key]
            for key in labels:
                labels[key][i] = label[key]

        return dataset, labels

    def test(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return the test set."""
        return self.smiles_to_dataset(self._test_smiles)

    def train(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return the training set."""
        return self.smiles_to_dataset(self._training_smiles)

    def train_split(
        self,
    ) -> Iterator[
        Tuple[
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        ]
    ]:
        """Split the dataset into training and test sets."""
        for train_indices, validation_indices in tqdm(
            self._validation_splitter.split(
                np.arange(len(self._training_smiles)),
                [smiles.first_class_label for smiles in self._training_smiles],
            ),
            total=self._validation_splitter.get_n_splits(),
            desc="Training and validation splits",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="split",
        ):
            yield (
                self.smiles_to_dataset(
                    [self._training_smiles[i] for i in train_indices]
                ),
                self.smiles_to_dataset(
                    [self._training_smiles[i] for i in validation_indices]
                ),
            )
