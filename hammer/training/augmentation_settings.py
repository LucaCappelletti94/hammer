"""Submodule handling the configuration of the augmentation settings for the dataset."""

from typing import Dict, List, Tuple, Type, Iterator, Optional
import numpy as np
from dict_hash import Hashable, sha256
from hammer.training.augmentation_strategies import (
    AugmentationStrategy,
    PickaxeAugmentationStrategy,
    StereoisomersAugmentationStrategy,
    TautomersAugmentationStrategy,
)

STRATEGIES: List[Type[AugmentationStrategy]] = [
    PickaxeAugmentationStrategy,
    StereoisomersAugmentationStrategy,
    TautomersAugmentationStrategy,
]


class AugmentationSettings(Hashable):
    """Class defining the augmentation settings for the dataset."""

    def __init__(self):
        """Initialize the augmentation settings."""
        self._augmentations: Dict[str, int] = {}

        for strategy_class in STRATEGIES:
            # We initialize all augmentations as not included
            self._augmentations[strategy_class.pythonic_name()] = 0

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return consistent hash of the current object."""
        return sha256(
            {
                "augmentations": self._augmentations,
            },
            use_approximation=use_approximation,
        )

    def _include_augmentation(self, augmentation_name: str, number_of_molecules: int):
        """Include a specific augmentation."""
        self._augmentations[augmentation_name] = number_of_molecules
        return self

    def _includes_augmentation(self, augmentation_name: str) -> bool:
        """Check if a specific augmentation is included."""
        return self._augmentations[augmentation_name] > 0

    def to_dict(self) -> Dict[str, int]:
        """Convert the settings to a dictionary."""
        return self._augmentations

    def includes_augmentations(self) -> bool:
        """Check if any augmentations are included."""
        return any(self._augmentations.values())

    def __getattr__(self, name):
        # If the attribute name starts with 'include_', handle it dynamically
        if name.startswith("include_"):
            augmentation_name = name[len("include_") :]
            if augmentation_name in self._augmentations:
                return lambda value: self._include_augmentation(
                    augmentation_name, value
                )

        if name.startswith("includes_"):
            augmentation_name = name[len("includes_") :]
            if augmentation_name in self._augmentations:
                return lambda: self._includes_augmentation(augmentation_name)

        if name.startswith("maximal_number_of_"):
            augmentation_name = name[len("maximal_number_of_") :]
            if augmentation_name in self._augmentations:
                return self._augmentations[augmentation_name]

        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute '{name}'"
        )

    @staticmethod
    def default() -> "AugmentationSettings":
        """Get the default augmentation settings."""
        return (
            AugmentationSettings()
            .include_tautomers(16)
            .include_stereoisomers(16)
            .include_pickaxe(64)
        )

    def iter_augmentations(self) -> Iterator[Type[AugmentationStrategy]]:
        """Iterate over the included augmentations."""
        for strategy_class in STRATEGIES:
            if self._includes_augmentation(strategy_class.pythonic_name()):
                yield strategy_class

    def augment(
        self,
        smiles: List[str],
        labels: Dict[str, np.ndarray],
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Augment the dataset."""
        if not self.includes_augmentations():
            return smiles, labels

        all_augmented_smiles: List[List[List[str]]] = [
            strategy_class(
                maximal_number=self._augmentations[strategy_class.pythonic_name()],
                n_jobs=n_jobs,
                verbose=verbose,
            ).augment_all(smiles)
            for strategy_class in self.iter_augmentations()
        ]

        # Flatten the list of lists, ensuring no duplicates
        flattened_augmented_smiles: List[List[str]] = [[s] for s in smiles]

        for augmentation_strategy_smiles in all_augmented_smiles:
            assert len(augmentation_strategy_smiles) == len(smiles)
            for agg, augmented_smiles in zip(
                flattened_augmented_smiles, augmentation_strategy_smiles
            ):
                agg.extend(augmented_smiles)

        # We ensure that the smiles in each sublist are unique,
        # and we sort them to ensure reproducibility
        flattened_augmented_smiles = [
            sorted(list(set(sublist))) for sublist in flattened_augmented_smiles
        ]

        # Finally, we flatten the list of lists and create also the extended labels
        total_number_of_smiles = sum(
            len(sublist) for sublist in flattened_augmented_smiles
        )

        extended_labels = {
            key: np.zeros((total_number_of_smiles, value.shape[1]), dtype=value.dtype)
            for key, value in labels.items()
        }

        augmented_smiles: List[str] = [
            smiles for sublist in flattened_augmented_smiles for smiles in sublist
        ]

        for labels_name, sub_labels in labels.items():
            for i, smiles in enumerate(flattened_augmented_smiles):
                extended_labels[labels_name][
                    i * len(smiles) : (i + 1) * len(smiles)
                ] = sub_labels[i]

        return augmented_smiles, extended_labels
