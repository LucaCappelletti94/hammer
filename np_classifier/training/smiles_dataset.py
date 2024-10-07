"""Submodule providing a class handling the SMILES dataset for training."""

from typing import List, Iterator, Tuple, Dict, Set, Optional, Type, Any
import os
import random
from collections import Counter
from multiprocessing import Pool
import compress_json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler
from cache_decorator import Cache
from np_classifier.training.molecular_features import compute_features
from np_classifier.training.augmentation_strategies import (
    StereoisomersAugmentationStrategy,
    TautomersAugmentationStrategy,
    AugmentationStrategy,
)


def _get_features(smiles_and_kwargs: (str, Dict[str, Any])) -> Dict[str, np.ndarray]:
    """Return the fingerprints of a Molecule."""
    smiles, kwargs = smiles_and_kwargs

    assert any(param for param in kwargs.values()), "No features are selected."

    x = compute_features(smiles, **kwargs)

    assert len(x) == sum(
        int(value) for value in kwargs.values() if isinstance(value, bool)
    ), "Incorrect number of features."

    return x


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
        include_morgan_fingerprint: bool = False,
        include_rdkit_fingerprint: bool = False,
        include_atom_pair_fingerprint: bool = False,
        include_topological_torsion_fingerprint: bool = False,
        include_avalon_fingerprint: bool = False,
        include_maccs_fingerprint: bool = False,
        include_map4_fingerprint: bool = False,
        include_descriptors: bool = False,
        include_skfp_autocorr_fingerprint: bool = False,
        include_skfp_ecfp_fingerprint: bool = False,
        include_skfp_erg_fingerprint: bool = False,
        include_skfp_estate_fingerprint: bool = False,
        include_skfp_functional_groups_fingerprint: bool = False,
        include_skfp_ghose_crippen_fingerprint: bool = False,
        include_skfp_klekota_roth_fingerprint: bool = False,
        include_skfp_laggner_fingerprint: bool = False,
        include_skfp_layered_fingerprint: bool = False,
        include_skfp_lingo_fingerprint: bool = False,
        include_skfp_map_fingerprint: bool = False,
        include_skfp_mhfp_fingerprint: bool = False,
        include_skfp_mqns_fingerprint: bool = False,
        include_skfp_pattern_fingerprint: bool = False,
        include_skfp_pubchem_fingerprint: bool = False,
        include_skfp_secfp_fingerprint: bool = False,
        include_skfp_vsa_fingerprint: bool = False,
        use_tautomer_augmentation_strategy: bool = False,
        maximal_number_of_tautomers: Optional[int] = 16,
        use_stereoisomer_augmentation_strategy: bool = True,
        maximal_number_of_stereoisomers: Optional[int] = 16,
        use_pickaxe_augmentation_strategy: bool = True,
        maximal_number_of_pickaxe_molecules: Optional[int] = 64,
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
        self._features_kwargs = {
            "radius": radius,
            "n_bits": n_bits,
            "include_morgan_fingerprint": include_morgan_fingerprint,
            "include_rdkit_fingerprint": include_rdkit_fingerprint,
            "include_atom_pair_fingerprint": include_atom_pair_fingerprint,
            "include_topological_torsion_fingerprint": include_topological_torsion_fingerprint,
            "include_avalon_fingerprint": include_avalon_fingerprint,
            "include_maccs_fingerprint": include_maccs_fingerprint,
            "include_map4_fingerprint": include_map4_fingerprint,
            "include_skfp_autocorr_fingerprint": include_skfp_autocorr_fingerprint,
            "include_skfp_ecfp_fingerprint": include_skfp_ecfp_fingerprint,
            "include_skfp_erg_fingerprint": include_skfp_erg_fingerprint,
            "include_skfp_estate_fingerprint": include_skfp_estate_fingerprint,
            "include_skfp_functional_groups_fingerprint": include_skfp_functional_groups_fingerprint,
            "include_skfp_ghose_crippen_fingerprint": include_skfp_ghose_crippen_fingerprint,
            "include_skfp_klekota_roth_fingerprint": include_skfp_klekota_roth_fingerprint,
            "include_skfp_laggner_fingerprint": include_skfp_laggner_fingerprint,
            "include_skfp_layered_fingerprint": include_skfp_layered_fingerprint,
            "include_skfp_lingo_fingerprint": include_skfp_lingo_fingerprint,
            "include_skfp_map_fingerprint": include_skfp_map_fingerprint,
            "include_skfp_mhfp_fingerprint": include_skfp_mhfp_fingerprint,
            "include_skfp_mqns_fingerprint": include_skfp_mqns_fingerprint,
            "include_skfp_pattern_fingerprint": include_skfp_pattern_fingerprint,
            "include_skfp_pubchem_fingerprint": include_skfp_pubchem_fingerprint,
            "include_skfp_secfp_fingerprint": include_skfp_secfp_fingerprint,
            "include_skfp_vsa_fingerprint": include_skfp_vsa_fingerprint,
            "include_descriptors": include_descriptors,
        }

        self._use_tautomer_augmentation_strategy = use_tautomer_augmentation_strategy
        self._use_stereoisomer_augmentation_strategy = (
            use_stereoisomer_augmentation_strategy
        )
        self._use_pickaxe_augmentation_strategy = use_pickaxe_augmentation_strategy
        self._maximal_number_of_tautomers = maximal_number_of_tautomers
        self._maximal_number_of_stereoisomers = maximal_number_of_stereoisomers
        self._maximal_number_of_pickaxe_molecules = maximal_number_of_pickaxe_molecules

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

    def augment_smiles(self, smiles: List[str]) -> List[List[str]]:
        """Returns the molecules augmented using the augmentation strategies."""
        augmented_smiles: Optional[List[List[str]]] = None

        if not any(
            [
                self._use_tautomer_augmentation_strategy,
                self._use_stereoisomer_augmentation_strategy,
                self._use_pickaxe_augmentation_strategy,
            ]
        ):
            return [[] for _ in smiles]

        augmentation_strategies: List[Type[AugmentationStrategy]] = []

        if self._use_tautomer_augmentation_strategy:
            augmentation_strategies.append(
                TautomersAugmentationStrategy(self._maximal_number_of_tautomers),
            )

        if self._use_stereoisomer_augmentation_strategy:
            augmentation_strategies.append(
                StereoisomersAugmentationStrategy(
                    self._maximal_number_of_stereoisomers
                ),
            )

        for strategy in tqdm(
            augmentation_strategies,
            desc="Augmenting molecules",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
        ):
            with Pool() as pool:
                new_augmented_smiles: List[List[str]] = list(
                    tqdm(
                        pool.imap(
                            strategy.augment,
                            smiles,
                        ),
                        desc=f"Augmenting using strategy '{strategy.name()}'",
                        leave=False,
                        dynamic_ncols=True,
                        disable=not self._verbose,
                        total=len(smiles),
                        unit="molecule",
                    )
                )
                pool.close()
                pool.join()

            if augmented_smiles is None:
                augmented_smiles = new_augmented_smiles
            else:
                for i, new_augmented in enumerate(
                    tqdm(
                        new_augmented_smiles,
                        desc=f"Extending smiles using strategy '{strategy.name()}'",
                        leave=False,
                        dynamic_ncols=True,
                        disable=not self._verbose,
                    )
                ):
                    augmented_smiles[i].extend(
                        [
                            smile
                            for smile in new_augmented
                            if smile not in augmented_smiles[i]
                        ]
                    )

        # For the pickaxe strategy, we need to use the precomputed reference dataset.
        if self._use_pickaxe_augmentation_strategy:
            pickaxe_dataset: Dict[str, List[str]] = compress_json.local_load(
                "pickaxe_normalized.json.xz"
            )

            new_augmented_smiles: List[List[str]] = []
            for smile in tqdm(
                smiles,
                desc="Augmenting using strategy 'pickaxe'",
                leave=False,
                dynamic_ncols=True,
                disable=not self._verbose,
            ):
                if smile in pickaxe_dataset:
                    associated_smiles: List[str] = pickaxe_dataset[smile]
                    if (
                        len(associated_smiles)
                        > self._maximal_number_of_pickaxe_molecules
                    ):
                        new_augmented_smiles.append(
                            random.choices(
                                associated_smiles,
                                k=self._maximal_number_of_pickaxe_molecules,
                            )
                        )
                    else:
                        new_augmented_smiles.append(associated_smiles)
                else:
                    new_augmented_smiles.append([])

            if augmented_smiles is None:
                augmented_smiles = new_augmented_smiles
            else:
                for i, new_augmented in enumerate(
                    tqdm(
                        new_augmented_smiles,
                        desc="Extending smiles using strategy 'pickaxe'",
                        leave=False,
                        dynamic_ncols=True,
                        disable=not self._verbose,
                    )
                ):
                    augmented_smiles[i].extend(
                        [
                            smile
                            for smile in new_augmented
                            if smile not in augmented_smiles[i]
                        ]
                    )

        # We deduplicate the augmented molecules.
        all_smiles: Set[str] = set(smiles)
        for i, smiles_list in enumerate(
            tqdm(
                augmented_smiles,
                desc="Deduplicating augmented smiles",
                leave=False,
                dynamic_ncols=True,
                disable=not self._verbose,
            )
        ):
            augmented_smiles[i] = [
                smile for smile in set(smiles_list) if smile not in all_smiles
            ]
            all_smiles.update(augmented_smiles[i])

        return augmented_smiles

    @Cache(
        use_approximated_hash=True,
    )
    def to_dataset(
        self,
        smiles: List[str],
        smiles_label_indices: List[int],
        scalers: Optional[Dict[str, RobustScaler]] = None,
    ) -> Tuple[
        Dict[str, RobustScaler], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ]:
        """Convert a list of Molecules to a dataset.

        Parameters
        ----------
        smiles : List[str]
            The SMILES of the molecules.
        smiles_label_indices : List[int]
            The indices of the labels of the molecules.
            While in the case of test or validation smiles these
            indices will be unique, in the case of training smiles
            when the smiles are augmented, the indices will be repeated.
        scalers : Optional[RobustScaler]
            The scaler to use to scale the features.
            If None, a new scaler is created.
        """
        number_of_molecules = len(smiles)
        # We compute the features of the first molecule to determine the shape of the features.
        first_smile_features = compute_features(smiles[0], **self._features_kwargs)

        assert len(first_smile_features) > 0, "No features are computed."
        expected_number_of_features = sum(
            int(value)
            for value in self._features_kwargs.values()
            if isinstance(value, bool)
        )
        assert (
            len(first_smile_features) == expected_number_of_features
        ), f"Incorrect number of features {len(first_smile_features)}, expected {expected_number_of_features}"

        # We allocate the arrays to store the features.
        features = {
            key: np.zeros((number_of_molecules, value.shape[0]), dtype=value.dtype)
            for key, value in first_smile_features.items()
        }

        with Pool() as pool:
            for i, smile_features in enumerate(
                tqdm(
                    pool.imap(
                        _get_features,
                        ((s, self._features_kwargs) for s in smiles),
                    ),
                    desc="Computing molecular features",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not self._verbose,
                    total=number_of_molecules,
                    unit="molecule",
                )
            ):
                for key in features:
                    features[key][i] = smile_features[key]
            pool.close()
            pool.join()

        # Now we allocate and fill the labels.
        labels = {
            "pathway": np.zeros(
                (number_of_molecules, len(self._pathway_names)), dtype=np.uint8
            ),
            "superclass": np.zeros(
                (number_of_molecules, len(self._superclass_names)), dtype=np.uint8
            ),
            "class": np.zeros(
                (number_of_molecules, len(self._class_names)), dtype=np.uint8
            ),
        }

        for i, label_index in enumerate(smiles_label_indices):
            for pathway_index in self._pathway_indices[label_index]:
                labels["pathway"][i, pathway_index] = 1
            for superclass_index in self._superclass_indices[label_index]:
                labels["superclass"][i, superclass_index] = 1
            for class_index in self._class_indices[label_index]:
                labels["class"][i, class_index] = 1

        if scalers is None:
            scalers = {}

        for key in features:
            if "fingerprint" in key:
                continue
            if key not in scalers:
                scalers[key] = RobustScaler().fit(features[key])

            features[key] = scalers[key].transform(features[key])

        return (scalers, (features, labels))

    def primary_split(
        self, augment: bool = True
    ) -> Tuple[RobustScaler, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Split the dataset into training and test sets."""
        if augment:
            training_smiles: List[str] = [self._smiles[i] for i in self._train_indices]
            augmented_training_smile: List[List[str]] = self.augment_smiles(
                training_smiles
            )
            training_molecules: List[str] = training_smiles + [
                smile for smiles in augmented_training_smile for smile in smiles
            ]
            original_label_indices: List[int] = [
                self._train_indices[i] for i in range(len(training_smiles))
            ]
            augmented_label_indices: List[int] = [
                self._train_indices[i]
                for i, augmented in zip(
                    range(len(training_smiles)), augmented_training_smile
                )
                for _ in range(len(augmented))
            ]
            training_label_indices: List[int] = (
                original_label_indices + augmented_label_indices
            )
        else:
            training_molecules: List[str] = [
                self._smiles[i] for i in self._train_indices
            ]
            training_label_indices: List[int] = [
                self._train_indices[i] for i in range(len(training_molecules))
            ]
        scalers, training_dataset = self.to_dataset(
            training_molecules, training_label_indices
        )

        _scalers, test_dataset = self.to_dataset(
            [self._smiles[i] for i in self._test_indices],
            self._test_indices,
            scalers=scalers,
        )
        return (scalers, training_dataset, test_dataset)

    def train_split(
        self,
        augment: bool = True,
    ) -> Iterator[
        Tuple[
            RobustScaler,
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        ]
    ]:
        """Split the dataset into training and test sets.

        Parameters
        ----------
        augment : bool
            Whether to augment the dataset.
        """
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
            if augment:
                training_smiles: List[str] = [
                    self._smiles[i] for i in self._train_indices[train_indices]
                ]
                augmented_training_smile: List[List[str]] = self.augment_smiles(
                    training_smiles
                )

                assert len(augmented_training_smile) == len(training_smiles)

                training_molecules: List[str] = training_smiles + [
                    smile for smiles in augmented_training_smile for smile in smiles
                ]
                original_label_indices: List[int] = [
                    self._train_indices[i] for i in train_indices
                ]
                augmented_label_indices: List[int] = [
                    self._train_indices[i]
                    for i, augmented in zip(train_indices, augmented_training_smile)
                    for _ in range(len(augmented))
                ]
                training_label_indices: List[int] = (
                    original_label_indices + augmented_label_indices
                )
            else:
                training_molecules: List[str] = [
                    self._smiles[i] for i in self._train_indices[train_indices]
                ]
                training_label_indices: List[int] = [
                    self._train_indices[i] for i in train_indices
                ]
            scalers, training_dataset = self.to_dataset(
                training_molecules, training_label_indices
            )
            _scalers, validation_dataset = self.to_dataset(
                [self._smiles[i] for i in self._train_indices[validation_indices]],
                [self._train_indices[i] for i in validation_indices],
                scalers=scalers,
            )
            yield (
                scalers,
                training_dataset,
                validation_dataset,
            )
