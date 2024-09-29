"""Submodule providing a class handling the SMILES dataset for training."""

from typing import List, Iterator, Tuple, Dict, Set
import os
from multiprocessing import Pool
import compress_json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rdkit.Chem import Mol  # pylint: disable=no-name-in-module
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from np_classifier.utils.constants import PATHWAY_NAMES, SUPERCLASS_NAMES, CLASS_NAMES
from np_classifier.training.molecule import Molecule
from np_classifier.training.augmentation_strategies import (
    generate_demethoxylated_homologues,
    generate_methoxylated_homologues,
)


def _get_fingerprints(molecule: (Molecule, int, int)) -> Dict[str, np.ndarray]:
    """Return the fingerprints of a Molecule."""
    molecule, radius, n_bits = molecule
    return molecule.fingerprint(radius=radius, n_bits=n_bits)


class Dataset:
    """Class handling the dataset for training."""

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
        molecules: List[Molecule] = [
            Molecule.from_smiles(
                smiles=row.smiles,
                pathway_labels=[row.pathway_label],
                superclass_labels=[row.superclass_label],
                class_labels=[row.class_label],
            )
            for row in pd.read_csv(categorical_smiles).itertuples()
        ] + [
            Molecule.from_smiles(
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
            np.arange(len(molecules)),
            stratify=[molecules.first_class_label for molecules in molecules],
            test_size=test_size,
            random_state=random_state,
        )

        self._test_molecules: List[Molecule] = [molecules[i] for i in test_indices]

        # We augment the training set.
        molecules_in_training_set: Set[Mol] = {
            molecules[i].molecule for i in train_indices
        }
        training_molecules: List[Molecule] = [molecules[i] for i in train_indices]
        augmented_molecules: List[Molecule] = []

        with Pool() as pool:
            demethoxylated_homologues: List[List[Mol]] = list(
                tqdm(
                    pool.imap(
                        generate_demethoxylated_homologues,
                        (molecule.molecule for molecule in training_molecules),
                    ),
                    desc="Generating demethoxylated homologues",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not verbose,
                    total=len(training_molecules),
                    unit="molecule",
                )
            )
            methoxylated_homologues: List[List[Mol]] = list(
                tqdm(
                    pool.imap(
                        generate_methoxylated_homologues,
                        (molecule.molecule for molecule in training_molecules),
                    ),
                    desc="Generating methoxylated homologues",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not verbose,
                    total=len(training_molecules),
                    unit="molecule",
                )
            )

        for i, molecule in enumerate(training_molecules):
            for homologue in demethoxylated_homologues[i] + methoxylated_homologues[i]:
                if homologue not in molecules_in_training_set:
                    augmented_molecules.append(molecule.into_homologue(homologue))
                    molecules_in_training_set.add(homologue)

        self._training_molecules: List[Molecule] = (
            training_molecules + augmented_molecules
        )
        self._validation_splitter = StratifiedShuffleSplit(
            n_splits=number_of_splits,
            test_size=validation_size,
            random_state=random_state // 2,
        )
        self._verbose = verbose
        self._radius = radius
        self._n_bits = n_bits

    def training_pathway_counts(self) -> Dict[str, int]:
        """Return the counts of the pathways in the training set."""
        counts = {pathway: 0 for pathway in PATHWAY_NAMES}
        for molecule in self._training_molecules:
            for pathway in molecule.pathway_labels:
                counts[PATHWAY_NAMES[pathway]] += 1
        return counts
    
    def training_superclass_counts(self) -> Dict[str, int]:
        """Return the counts of the superclasses in the training set."""
        counts = {superclass: 0 for superclass in SUPERCLASS_NAMES}
        for molecule in self._training_molecules:
            for superclass in molecule.superclass_labels:
                counts[SUPERCLASS_NAMES[superclass]] += 1
        return counts

    def training_class_counts(self) -> Dict[str, int]:
        """Return the counts of the classes in the training set."""
        counts = {class_: 0 for class_ in CLASS_NAMES}
        for molecule in self._training_molecules:
            for class_ in molecule.class_labels:
                counts[CLASS_NAMES[class_]] += 1
        return counts

    def to_dataset(
        self, molecules: List[Molecule]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Convert a list of Molecules to a dataset."""
        number_of_molecules = len(molecules)

        with Pool() as pool:
            fingerprints = list(
                tqdm(
                    pool.imap(
                        _get_fingerprints,
                        (
                            (molecule, self._radius, self._n_bits)
                            for molecule in molecules
                        ),
                    ),
                    desc="Generating fingerprints",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not self._verbose,
                    total=number_of_molecules,
                    unit="molecule",
                )
            )

        molecule_labels = molecules[0].labels()
        dataset = {
            key: np.zeros((number_of_molecules, value.shape[0]), dtype=np.float32)
            for key, value in fingerprints[0].items()
        }
        labels = {
            key: np.zeros((number_of_molecules, value.shape[0]), dtype=np.float32)
            for key, value in molecule_labels.items()
        }

        for i, (molecule, fingerprint) in enumerate(zip(molecules, fingerprints)):
            label = molecule.labels()
            for key in dataset:
                dataset[key][i] = fingerprint[key]
            for key in labels:
                labels[key][i] = label[key]

        return dataset, labels

    def test(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return the test set."""
        return self.to_dataset(self._test_molecules)

    def train(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return the training set."""
        return self.to_dataset(self._training_molecules)

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
                np.arange(len(self._training_molecules)),
                [molecule.first_class_label for molecule in self._training_molecules],
            ),
            total=self._validation_splitter.get_n_splits(),
            desc="Training and validation splits",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="split",
        ):
            yield (
                self.to_dataset([self._training_molecules[i] for i in train_indices]),
                self.to_dataset(
                    [self._training_molecules[i] for i in validation_indices]
                ),
            )
