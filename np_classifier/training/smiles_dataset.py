"""Submodule providing a class handling the SMILES dataset for training."""

from typing import List, Iterator, Tuple, Dict, Set, Optional, Type
import os
from multiprocessing import Pool
import compress_json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler
from np_classifier.utils.constants import PATHWAY_NAMES, SUPERCLASS_NAMES, CLASS_NAMES
from np_classifier.training.molecule import Molecule
from np_classifier.training.augmentation_strategies import (
    StereoisomersAugmentationStrategy,
    TautomersAugmentationStrategy,
    AugmentationStrategy,
)


def _get_features(molecule: (Molecule, int, int)) -> Dict[str, np.ndarray]:
    """Return the fingerprints of a Molecule."""
    molecule, radius, n_bits = molecule
    return molecule.features(radius=radius, n_bits=n_bits)


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
        maximal_number_of_molecules: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the SMILES dataset.

        Parameters
        ----------
        random_state : int
            The random state to use for the train/test/validation splits.
        number_of_splits : int
            The number of splits to use for the validation set.
        validation_size : float
            The size of the validation set.
        test_size : float
            The size of the test set.
        radius : int
            The radius of the fingerprint.
        n_bits : int
            The number of bits in the fingerprint.
        maximal_number_of_molecules : Optional[int]
            The maximal number of molecules to use.
            This is primarily used for testing.
            By default, all the molecules are used.
        verbose : bool
            Whether to display progress bars.
        """
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

        if maximal_number_of_molecules is not None:
            molecules = molecules[:maximal_number_of_molecules]

        # We determine a count of all the class labels in the dataset.
        class_counts = {class_: 0 for class_ in CLASS_NAMES}
        for molecule in molecules:
            for class_ in molecule.class_labels:
                class_counts[CLASS_NAMES[class_]] += 1

        if maximal_number_of_molecules is not None:
            molecules = [
                molecule
                for molecule in molecules
                if all(
                    class_counts[class_label] > 1
                    for class_label in molecule.class_label_names
                )
            ]

        # While it would be quite hard to stratify the dataset based on the combination
        # of pathway, superclass and class labels, we can achieve a reasonable stratification
        # by stratifying based on the least common label, which is the class label.
        train_indices, test_indices = train_test_split(
            np.arange(len(molecules)),
            stratify=[
                molecules.least_common_class_label(class_counts)
                for molecules in molecules
            ],
            test_size=test_size,
            random_state=random_state,
        )

        self._test_molecules: List[Molecule] = [molecules[i] for i in test_indices]
        self._training_molecules: List[Molecule] = [molecules[i] for i in train_indices]

        self._validation_splitter = StratifiedShuffleSplit(
            n_splits=number_of_splits,
            test_size=validation_size,
            random_state=random_state // 2 + 1,
        )
        self._verbose = verbose
        self._radius = radius
        self._n_bits = n_bits
        self._augmentation_strategies: List[Type[AugmentationStrategy]] = [
            # PickaxeStrategy(),
            TautomersAugmentationStrategy(),
            StereoisomersAugmentationStrategy(),
        ]

    def augment_molecules(self, original_molecules: List[Molecule]) -> List[Molecule]:
        """Returns the molecules augmented using the augmentation strategies."""
        augmented_molecules: Set[Molecule] = set(original_molecules)

        for strategy in tqdm(
            self._augmentation_strategies,
            desc="Augmenting molecules",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
        ):
            with Pool() as pool:
                new_molecules: List[List[Molecule]] = list(
                    tqdm(
                        pool.imap(
                            strategy.augment,
                            original_molecules,
                        ),
                        desc=f"Augmenting using strategy '{strategy.name()}'",
                        leave=False,
                        dynamic_ncols=True,
                        disable=not self._verbose,
                        total=len(original_molecules),
                        unit="molecule",
                    )
                )
                pool.close()
                pool.join()

            for molecules in new_molecules:
                augmented_molecules.update(molecules)

        return list(augmented_molecules)

    @property
    def training_molecules(self) -> List[Molecule]:
        """Return the training set."""
        return self._training_molecules

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
        self,
        molecules: List[Molecule],
        scalers: Optional[Dict[str, RobustScaler]] = None,
    ) -> Tuple[
        Dict[str, RobustScaler], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ]:
        """Convert a list of Molecules to a dataset.

        Parameters
        ----------
        molecules : List[Molecule]
            The molecules to convert to a dataset.
        scalers : Optional[RobustScaler]
            The scaler to use to scale the features.
            If None, a new scaler is created.
        """
        number_of_molecules = len(molecules)

        with Pool() as pool:
            features = list(
                tqdm(
                    pool.imap(
                        _get_features,
                        (
                            (molecule, self._radius, self._n_bits)
                            for molecule in molecules
                        ),
                    ),
                    desc="Computing molecular features",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not self._verbose,
                    total=number_of_molecules,
                    unit="molecule",
                )
            )
            pool.close()
            pool.join()

        molecule_labels = molecules[0].labels()
        dataset = {
            key: np.zeros((number_of_molecules, value.shape[0]), dtype=np.float32)
            for key, value in features[0].items()
        }
        labels = {
            key: np.zeros((number_of_molecules, value.shape[0]), dtype=np.float32)
            for key, value in molecule_labels.items()
        }

        for i, (molecule, fingerprint) in enumerate(zip(molecules, features)):
            label = molecule.labels()
            for key in dataset:
                dataset[key][i] = fingerprint[key]
            for key in labels:
                labels[key][i] = label[key]

        if scalers is None:
            scalers = {}

        for key in dataset:
            if "fingerprint" in key:
                continue
            if key not in scalers:
                scalers[key] = RobustScaler().fit(dataset[key])

            dataset[key] = scalers[key].transform(dataset[key])

        return (scalers, (dataset, labels))

    def primary_split(
        self,
    ) -> Tuple[
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    ]:
        """Return the training set."""
        augmented_training_molecules = self.augment_molecules(self._training_molecules)
        scalers, dataset = self.to_dataset(augmented_training_molecules)
        _, test_dataset = self.to_dataset(self._test_molecules, scalers=scalers)
        return (dataset, test_dataset)

    def train_features_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return the training set as dataframes."""
        (training_dataset, _) = self.to_dataset(self._training_molecules)
        descriptor_names = Molecule.descriptor_names()

        training_dataset["descriptors"] = pd.DataFrame(
            training_dataset["descriptors"], columns=descriptor_names
        )

        for key in training_dataset:
            if "fingerprint" in key:
                training_dataset[key] = pd.DataFrame(training_dataset[key], index=None)

        return training_dataset

    def train_split(
        self,
    ) -> Iterator[
        Tuple[
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        ]
    ]:
        """Split the dataset into training and test sets."""
        class_counts = self.training_class_counts()
        for train_indices, validation_indices in tqdm(
            self._validation_splitter.split(
                np.arange(len(self._training_molecules)),
                [
                    molecule.least_common_class_label(class_counts)
                    for molecule in self._training_molecules
                ],
            ),
            total=self._validation_splitter.get_n_splits(),
            desc="Training and validation splits",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="split",
        ):
            augmented_training_molecules = self.augment_molecules(
                [self._training_molecules[i] for i in train_indices]
            )
            scalers, training_dataset = self.to_dataset(augmented_training_molecules)
            _scalers, validation_dataset = self.to_dataset(
                [self._training_molecules[i] for i in validation_indices],
                scalers=scalers,
            )
            yield (
                training_dataset,
                validation_dataset,
            )
