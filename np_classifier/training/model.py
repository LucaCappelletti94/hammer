"""Module containing the multi-modal multi-class classifier model."""

from typing import Dict, Optional, Tuple, List
import os
from tensorflow.keras.models import (  # pylint: disable=no-name-in-module,import-error
    Model,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras.layers import (  # pylint: disable=no-name-in-module,import-error
    Concatenate,  # pylint: disable=no-name-in-module,import-error
    Layer,  # pylint: disable=no-name-in-module,import-error
    Input,  # pylint: disable=no-name-in-module,import-error
    Dense,  # pylint: disable=no-name-in-module,import-error
    BatchNormalization,  # pylint: disable=no-name-in-module,import-error
    LayerNormalization,  # pylint: disable=no-name-in-module,import-error
    Dropout,  # pylint: disable=no-name-in-module,import-error
    Multiply,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras import ops as K  # pylint: disable=no-name-in-module,import-error
from tensorflow.keras.utils import (  # pylint: disable=no-name-in-module,import-error
    plot_model,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras.callbacks import (  # pylint: disable=no-name-in-module,import-error
    ModelCheckpoint,  # pylint: disable=no-name-in-module,import-error
    TerminateOnNaN,  # pylint: disable=no-name-in-module,import-error
    ReduceLROnPlateau,  # pylint: disable=no-name-in-module,import-error
    EarlyStopping,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras.optimizers import (  # pylint: disable=no-name-in-module,import-error
    Adam,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras.initializers import (  # pylint: disable=no-name-in-module,import-error
    HeNormal,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras.saving import (  # pylint: disable=no-name-in-module,import-error
    load_model,  # pylint: disable=no-name-in-module,import-error
)
import tensorflow as tf
import compress_json
from downloaders import BaseDownloader
from tqdm.keras import TqdmCallback
import numpy as np
import pandas as pd
import compress_pickle
from sklearn.preprocessing import RobustScaler
from plot_keras_history import plot_history
from extra_keras_metrics import get_minimal_multiclass_metrics
from np_classifier.training.molecular_features import compute_features
from np_classifier.training.exceptions import (
    UnknownPathwayNameError,
    UnknownSuperclassNameError,
)


class Classifier:
    """Class representing the multi-modal multi-class classifier model."""

    def __init__(
        self,
        pathway_names: List[str],
        superclass_names: List[str],
        class_names: List[str],
        scalers: Dict[str, RobustScaler],
        model_path: Optional[str] = None,
    ):
        """Initialize the classifier model."""
        # Some healthy defensive programming.
        assert isinstance(pathway_names, list)
        assert isinstance(superclass_names, list)
        assert isinstance(class_names, list)
        assert isinstance(scalers, dict)
        assert len(pathway_names) > 0
        assert len(superclass_names) > 0
        assert len(class_names) > 0
        assert all(isinstance(name, str) for name in pathway_names)
        assert all(isinstance(name, str) for name in superclass_names)
        assert all(isinstance(name, str) for name in class_names)

        if model_path is not None:
            self._model = load_model(model_path)
        else:
            self._model: Optional[Model] = None
        self._history: Optional[pd.DataFrame] = None
        self._scalers: Dict[str, RobustScaler] = scalers
        self._pathway_names: List[str] = pathway_names
        self._superclass_names: List[str] = superclass_names
        self._class_names: List[str] = class_names

        dag = compress_json.local_load("dag.json")

        # We create a mask to harmonize the predictions.
        self._pathway_to_superclass_mask = np.zeros(
            (len(self._pathway_names), len(self._superclass_names)),
            dtype=np.uint8,
        )
        self._superclass_to_class_mask = np.zeros(
            (len(self._superclass_names), len(self._class_names)),
            dtype=np.uint8,
        )
        for i, superclass_name in enumerate(self._superclass_names):
            for pathway_name in dag["superclasses"][superclass_name]:
                try:
                    pathway_index = self._pathway_names.index(pathway_name)
                    self._pathway_to_superclass_mask[pathway_index, i] = 1
                except ValueError as index_error:
                    raise UnknownPathwayNameError(
                        pathway_name, superclass_name
                    ) from index_error
        for j, class_name in enumerate(self._class_names):
            for superclass_name in dag["classes"][class_name]:
                try:
                    superclass_index = self._superclass_names.index(superclass_name)
                    self._superclass_to_class_mask[superclass_index, j] = 1
                except ValueError as index_error:
                    raise UnknownSuperclassNameError(
                        superclass_name, class_name
                    ) from index_error

    @staticmethod
    def load(model_name: str) -> "Classifier":
        """Load a classifier model from a file."""
        all_model_url: Dict[str, str] = compress_json.local_load("models.json")
        if model_name not in all_model_url:
            raise ValueError(
                f"Model {model_name} not found. Available models: {all_model_url.keys()}"
            )

        # We download the model weights and metadata from Zenodo.
        downloader = BaseDownloader()
        path = f"downloads/{model_name}.tar.gz"
        downloader.download(urls=all_model_url[model_name], paths=path)

        class_names_path = f"downloads/{model_name}/class_names.json"
        pathway_names_path = f"downloads/{model_name}/pathway_names.json"
        superclass_names_path = f"downloads/{model_name}/superclass_names.json"
        scalers_path = f"downloads/{model_name}/scalers.pkl"
        model_path = f"downloads/{model_name}/model.keras"

        classifier = Classifier(
            class_names=compress_json.load(class_names_path),
            pathway_names=compress_json.load(pathway_names_path),
            superclass_names=compress_json.load(superclass_names_path),
            scalers=compress_pickle.load(scalers_path),
            model_path=model_path,
        )

        return classifier

    def predict_smile(
        self, smile: str, include_top_k: Optional[int] = 10
    ) -> Dict[str, str]:
        """Predict the class labels for a single SMILES string."""
        assert isinstance(smile, str)
        assert len(smile) > 0
        model_input_layer_names = list(self._model.input.keys())
        features: Dict[str, np.ndarray] = compute_features(
            smile,
            include_morgan_fingerprint="morgan_fingerprint" in model_input_layer_names,
            include_rdkit_fingerprint="rdkit_fingerprint" in model_input_layer_names,
            include_atom_pair_fingerprint="atom_pair_fingerprint"
            in model_input_layer_names,
            include_topological_torsion_fingerprint="topological_torsion_fingerprint"
            in model_input_layer_names,
            include_avalon_fingerprint="avalon_fingerprint" in model_input_layer_names,
            include_maccs_fingerprint="maccs_fingerprint" in model_input_layer_names,
            include_map4_fingerprint="map4_fingerprint" in model_input_layer_names,
            include_descriptors="descriptors" in model_input_layer_names,
        )

        features: Dict[str, np.ndarray] = {
            key: value.reshape(1, -1) for key, value in features.items()
        }

        # We scale the features using the scalers.
        assert self._scalers is not None
        for key, value in features.items():
            if key in self._scalers:
                features[key] = self._scalers[key].transform(value)

        predictions = self._model.predict(features)

        pathway_predictions = dict(zip(self._pathway_names, predictions["pathway"][0]))
        superclass_predictions = dict(
            zip(self._superclass_names, predictions["superclass"][0])
        )
        class_predictions = dict(zip(self._class_names, predictions["class"][0]))

        if include_top_k is not None:
            pathway_predictions = dict(
                sorted(
                    pathway_predictions.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:include_top_k]
            )
            superclass_predictions = dict(
                sorted(
                    superclass_predictions.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:include_top_k]
            )
            class_predictions = dict(
                sorted(
                    class_predictions.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:include_top_k]
            )

        return {
            "pathway": pathway_predictions,
            "superclass": superclass_predictions,
            "class": class_predictions,
        }

    def _build_input_modality(self, input_layer: Input) -> Layer:
        """Build the input modality sub-module."""
        hidden = input_layer

        if input_layer.shape[1] == 2048:
            hidden_sizes = 768
        else:
            hidden_sizes = 128

        for i in range(4):
            hidden = Dense(
                hidden_sizes,
                activation="relu",
                kernel_initializer=HeNormal(),
                name=f"dense_{input_layer.name}_{i}",
            )(hidden)
            hidden = LayerNormalization(
                name=f"layer_normalization_{input_layer.name}_{i}"
            )(hidden)
        hidden = Dropout(0.3)(hidden)

        return hidden

    def _build_hidden_layers(self, inputs: List[Layer]) -> Layer:
        """Build the hidden layers sub-module."""
        hidden = Concatenate(axis=-1)(inputs)
        for i in range(4):
            hidden = Dense(
                2048,
                activation="relu",
                kernel_initializer=HeNormal(),
                name=f"dense_hidden_{i}",
            )(hidden)
            hidden = LayerNormalization(
                name=f"layer_normalization_hidden_{i}",
            )(hidden)
        hidden = Dropout(0.3)(hidden)
        return hidden

    def _build_pathway_head(self, hidden_output: tf.Tensor) -> tf.Tensor:
        """Build the output head sub-module."""
        return Dense(len(self._pathway_names), name="pathway", activation="sigmoid")(
            hidden_output
        )

    def _build_superclass_head(
        self,
        hidden_output: tf.Tensor,
        pathway_output: tf.Tensor,
    ) -> tf.Tensor:
        """Build the output head sub-module."""
        unharmonized_superclass_output: tf.Tensor = Dense(
            len(self._superclass_names), name="unharmonized_superclass", activation="sigmoid"
        )(hidden_output)
        # We use the pathway output and the self._pathway_to_superclass_mask to compute
        # the harmonized superclass output. We do so by multiplying the pathway output
        # by the mask, summing the results over the superclasses axes, and then multiplying
        # the result by the superclass output.
        weighted_superclass_mask: tf.Tensor = K.sum(
            pathway_output[:, :, None] * self._pathway_to_superclass_mask[None, :, :],
            axis=1,
        )
        harmonized_superclass_output: tf.Tensor = Multiply(name="superclass")(
            [unharmonized_superclass_output, weighted_superclass_mask]
        )

        return harmonized_superclass_output

    def _build_class_head(
        self, hidden_output: tf.Tensor, superclasses_output: tf.Tensor
    ) -> tf.Tensor:
        """Build the output head sub-module."""
        unharmonized_class_output: tf.Tensor = Dense(
            len(self._class_names), name="unharmonized_class", activation="sigmoid"
        )(hidden_output)
        # We use the superclass output and the self._superclass_to_class_mask to compute
        # the harmonized class output. We do so by multiplying the superclass output
        # by the mask, summing the results over the classes axes, and then multiplying
        # the result by the class output.
        weighted_class_mask: tf.Tensor = K.sum(
            superclasses_output[:, :, None]
            * self._superclass_to_class_mask[None, :, :],
            axis=1,
        )
        harmonized_class_output: tf.Tensor = Multiply(name="class")(
            [unharmonized_class_output, weighted_class_mask]
        )

        return harmonized_class_output

    def _build(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
    ):
        """Build the classifier model."""
        # Validate the input types.
        assert isinstance(inputs, dict)
        assert isinstance(outputs, dict)
        assert all(isinstance(value, np.ndarray) for value in inputs.values())
        assert all(isinstance(value, np.ndarray) for value in outputs.values())

        # Validate the output shapes.
        assert outputs["pathway"].shape[1] == len(self._pathway_names)
        assert outputs["superclass"].shape[1] == len(self._superclass_names)
        assert outputs["class"].shape[1] == len(self._class_names)

        input_layers: List[Input] = [
            Input(shape=input_array.shape[1:], name=name, dtype=input_array.dtype)
            for name, input_array in inputs.items()
        ]

        input_modalities: List[Layer] = [
            self._build_input_modality(input_layer) for input_layer in input_layers
        ]

        hidden: Layer = self._build_hidden_layers(input_modalities)

        pathway_head = self._build_pathway_head(hidden_output=hidden)
        superclass_head = self._build_superclass_head(
            hidden_output=hidden,
            pathway_output=pathway_head,
        )
        class_head = self._build_class_head(
            hidden_output=hidden, superclasses_output=superclass_head
        )

        self._model = Model(
            inputs={input_layer.name: input_layer for input_layer in input_layers},
            outputs={
                "pathway": pathway_head,
                "superclass": superclass_head,
                "class": class_head,
            },
            name="classifier",
        )

    def train(
        self,
        train: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        val: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        holdout_number: Optional[int] = None,
        number_of_epochs: int = 10_000,
    ):
        """Train the classifier model."""
        self._build(*train)
        self._model.compile(
            optimizer=Adam(),
            loss="binary_focal_crossentropy",
            metrics={
                "pathway": get_minimal_multiclass_metrics(),
                "superclass": get_minimal_multiclass_metrics(),
                "class": get_minimal_multiclass_metrics(),
            },
        )
        plot_model(
            self._model,
            to_file="model.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=100,
            show_layer_activations=True,
            show_trainable=True,
        )

        if holdout_number is not None:
            model_checkpoint_path = f"model_checkpoint_{holdout_number}.keras"
        else:
            model_checkpoint_path = "model_checkpoint.keras"

        model_checkpoint = ModelCheckpoint(
            model_checkpoint_path,
            monitor="val_class_AUPRC",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            verbose=0,
        )

        learning_rate_scheduler = ReduceLROnPlateau(
            monitor="val_class_AUPRC",
            factor=0.8,
            patience=20,
            verbose=1,
            mode="max",
            min_delta=1e-4,
            cooldown=5,
            min_lr=1e-6,
        )

        early_stopping = EarlyStopping(
            monitor="val_class_AUPRC",
            patience=100,
            verbose=1,
            mode="max",
            restore_best_weights=True,
        )

        training_history = self._model.fit(
            *train,
            epochs=number_of_epochs,
            callbacks=[
                TqdmCallback(
                    verbose=1,
                    metrics=[
                        "loss",
                        "val_loss",
                        "class_AUPRC",
                        "val_class_AUPRC",
                        "superclass_AUPRC",
                        "val_superclass_AUPRC",
                        "pathway_AUPRC",
                        "val_pathway_AUPRC",
                    ],
                ),
                model_checkpoint,
                TerminateOnNaN(),
                early_stopping,
                learning_rate_scheduler,
            ],
            batch_size=4096,
            shuffle=True,
            verbose=0,
            validation_data=val,
        )
        self._history = pd.DataFrame(training_history.history)

        fig, _ = plot_history(
            self._history, monitor="val_class_AUPRC", monitor_mode="max"
        )

        # We create a directory 'histories' if it does not exist.
        os.makedirs("histories", exist_ok=True)

        if holdout_number is not None:
            self._history.to_csv(f"histories/history_{holdout_number}.csv")
            fig.savefig(f"histories/history_{holdout_number}.png")
        else:
            self._history.to_csv("histories/history.csv")
            fig.savefig("histories/history.png")

    def save(self, path: str):
        """Save the classifier model to a file."""
        assert self._model is not None, "Model not trained yet."
        assert path.endswith(".tar.gz"), "Model path must end with .tar.gz"

        # We convert the path into a directory.
        path = path[:-7]
        # We create the directories if they do not exist.
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "model.keras")
        class_names_path = os.path.join(path, "class_names.json")
        pathway_names_path = os.path.join(path, "pathway_names.json")
        superclass_names_path = os.path.join(path, "superclass_names.json")
        scalers_path = os.path.join(path, "scalers.pkl")

        compress_json.dump(self._class_names, class_names_path)
        compress_json.dump(self._pathway_names, pathway_names_path)
        compress_json.dump(self._superclass_names, superclass_names_path)
        compress_pickle.dump(self._scalers, scalers_path)
        self._model.save(model_path)

        # We compress the files into a single tar.gz file.
        os.system(f"tar -czvf {path}.tar.gz {path}")

    def evaluate(
        self, test: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Evaluate the classifier model."""
        return self._model.evaluate(*test, verbose=0, return_dict=True)
