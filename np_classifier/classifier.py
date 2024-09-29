import os
from typing import List, Dict
import numpy as np

from downloaders import BaseDownloader
import compress_json
from tensorflow import keras  # pylint: disable=no-name-in-module
from rdkit.Chem import Mol  # pylint: disable=no-name-in-module
from np_classifier.utils import (
    to_morgan_fingerprint,
    as_list,
    as_one_hot,
)


class Classification:
    """Class to classify natural products."""

    def __init__(
        self,
        pathway_scores: np.ndarray,
        superclass_scores: np.ndarray,
        class_scores: np.ndarray,
    ):
        dictionary = compress_json.local_load("classifications.json")

        self._pathways: List[str] = as_list(dictionary["Pathway"])
        self._superclasses: List[str] = as_list(dictionary["Superclass"])
        self._classes: List[str] = as_list(dictionary["Class"])

        self._pathway_scores: np.ndarray = pathway_scores
        self._superclass_scores: np.ndarray = superclass_scores
        self._class_scores: np.ndarray = class_scores

    def top_k_pathways(self, k: int) -> Dict[str, float]:
        """Returns the top k pathways."""
        indices = np.argsort(self._pathway_scores)[::-1][:k]
        return {self._pathways[i]: self._pathway_scores[i] for i in indices}

    def top_k_superclasses(self, k: int) -> Dict[str, float]:
        """Returns the top k superclasses."""
        indices = np.argsort(self._superclass_scores)[::-1][:k]
        return {self._superclasses[i]: self._superclass_scores[i] for i in indices}

    def top_k_classes(self, k: int) -> Dict[str, float]:
        """Returns the top k classes."""
        indices = np.argsort(self._class_scores)[::-1][:k]
        return {self._classes[i]: self._class_scores[i] for i in indices}


class NPClassifier:
    """Class to classify natural products."""

    def __init__(self):
        downloader = BaseDownloader()
        downloader.download(
            "https://zenodo.org/record/5068687/files/model.zip?download=1", "model.zip"
        )
        dictionary = compress_json.local_load("classifications.json")

        self._pathways: List[str] = as_list(dictionary["Pathway"])
        self._superclasses: List[str] = as_list(dictionary["Superclass"])
        self._classes: List[str] = as_list(dictionary["Class"])

        superclass_to_pathways = [[] for _ in range(len(self._superclasses))]
        for superclass_str_index, pathway_data in dictionary["Super_hierarchy"].items():
            superclass_index: int = int(superclass_str_index)
            for pathway_index in pathway_data["Pathway"]:
                superclass_to_pathways[superclass_index].append(pathway_index)

        self._superclass_to_pathways: List[List[int]] = superclass_to_pathways

        class_to_superclasses = [[] for _ in range(len(self._classes))]
        class_to_pathways = [[] for _ in range(len(self._classes))]

        for class_str_index, pathway_data in dictionary["Class_hierarchy"].items():
            class_index: int = int(class_str_index)
            for pathway_index in pathway_data["Pathway"]:
                class_to_pathways[class_index].append(pathway_index)
            for superclass_index in pathway_data["Superclass"]:
                class_to_superclasses[class_index].append(superclass_index)

        self._class_to_pathways: List[List[int]] = class_to_pathways
        self._class_to_superclasses: List[List[int]] = class_to_superclasses

        assert os.path.exists(
            "model/NP_classifier_class_V1.hdf5"
        ), "Could not find Class classifier"
        assert os.path.exists(
            "model/NP_classifier_superclass_V1.hdf5"
        ), "Could not find Superclass classifier"
        assert os.path.exists(
            "model/NP_classifier_pathway_V1.hdf5"
        ), "Could not find Pathway classifier"

        self._model_class = keras.models.load_model("model/NP_classifier_class_V1.hdf5")
        self._model_super = keras.models.load_model(
            "model/NP_classifier_superclass_V1.hdf5"
        )
        self._model_pathway = keras.models.load_model(
            "model/NP_classifier_pathway_V1.hdf5"
        )

    @property
    def pathways(self) -> List[str]:
        """Returns the list of pathways."""
        return self._pathways

    @property
    def number_of_pathways(self) -> int:
        """Returns the number of pathways."""
        return len(self._pathways)

    @property
    def superclasses(self) -> List[str]:
        """Returns the list of superclasses."""
        return self._superclasses

    @property
    def number_of_superclasses(self) -> int:
        """Returns the number of superclasses."""
        return len(self._superclasses)

    @property
    def classes(self) -> List[str]:
        """Returns the list of classes."""
        return self._classes

    @property
    def number_of_classes(self) -> int:
        """Returns the number of classes."""
        return len(self._classes)

    def classify(self, molecule: Mol) -> Classification:
        """Classifies a natural product given its SMILES string."""

        smiles_fingerprint = to_morgan_fingerprint(molecule, radius=2, n_bits=2048)

        pathway_predictions = self._model_pathway.predict(smiles_fingerprint)[0]
        superclass_predictions = self._model_super.predict(smiles_fingerprint)[0]
        class_predictions = self._model_class.predict(smiles_fingerprint)[0]

        pathway_predictions_by_superclass = np.mean(
            [
                superclass_prediction
                * as_one_hot(superclass_pathways, self.number_of_pathways)
                for superclass_prediction, superclass_pathways in zip(
                    superclass_predictions, self._superclass_to_pathways
                )
            ]
        )

        pathway_predictions_by_class = np.mean(
            [
                class_prediction * as_one_hot(class_pathways, self.number_of_pathways)
                for class_prediction, class_pathways in zip(
                    class_predictions, self._class_to_pathways
                )
            ]
        )

        pathway_predictions_by_class = pathway_predictions_by_class / np.sum(
            pathway_predictions_by_class
        )

        superclass_predictions_by_class = np.mean(
            [
                class_prediction
                * as_one_hot(class_superclasses, self.number_of_superclasses)
                for class_prediction, class_superclasses in zip(
                    class_predictions, self._class_to_superclasses
                )
            ]
        )

        superclass_predictions_by_class = superclass_predictions_by_class / np.sum(
            superclass_predictions
        )

        pathway_predictions = (
            pathway_predictions
            * pathway_predictions_by_superclass
            * pathway_predictions_by_class
        )
        superclass_predictions = (
            superclass_predictions * superclass_predictions_by_class
        )

        pathway_predictions = pathway_predictions / np.sum(pathway_predictions)
        superclass_predictions = superclass_predictions / np.sum(superclass_predictions)

        return Classification(
            pathway_predictions, superclass_predictions, class_predictions
        )
