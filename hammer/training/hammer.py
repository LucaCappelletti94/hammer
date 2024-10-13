"""Module containing the hierarchical multi-modal multi-class classifier model."""

from typing import Dict, Optional, Any, List, Type
import os
import shutil
from tensorflow.keras.models import (  # pylint: disable=no-name-in-module,import-error
    Model,  # pylint: disable=no-name-in-module,import-error
)
from tensorflow.keras.layers import (  # pylint: disable=no-name-in-module,import-error
    Concatenate,  # pylint: disable=no-name-in-module,import-error
    Layer,  # pylint: disable=no-name-in-module,import-error
    Input,  # pylint: disable=no-name-in-module,import-error
    Dense,  # pylint: disable=no-name-in-module,import-error
    BatchNormalization,  # pylint: disable=no-name-in-module,import-error
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
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import compress_pickle
from sklearn.preprocessing import RobustScaler
from plot_keras_history import plot_history
from extra_keras_metrics import get_minimal_multiclass_metrics
from hammer.training.feature_settings import FeatureSettings
from hammer.training.features import FeatureInterface
from hammer.training.augmentation_settings import AugmentationSettings
from hammer.training.utils import into_canonical_smiles, smiles_to_molecules
from hammer.training.layered_dags import LayeredDAG


class Hammer:
    """Hierarchical Augmented Multi-modal Multi-class classifiER."""

    def __init__(
        self,
        dag: Type[LayeredDAG],
        feature_settings: FeatureSettings,
        scalers: Optional[Dict[str, RobustScaler]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ):
        """Initialize the classifier model.

        Parameters
        ----------
        label_names : Dict[str, List[str]]
            Dictionary of label names, with the keys being the label categories.
        dag : Dict[str, Dict[str, List[str]]]
            Directed acyclic graph defining the relationships between the labels.
        feature_settings : FeatureSettings
            Feature settings used to compute the features.
        scalers : Optional[Dict[str, RobustScaler]]
            Dictionary of scalers used to scale the features.
        metadata : Optional[Dict[str, Any]]
            Metadata of the classifier model.
            Used to store additional information about the classifier.
        model_path : Optional[str]
            Path to the classifier model, if it has been trained.
        verbose : bool
            Whether to display a progress bar.
        n_jobs : Optional[int]
            The number of jobs to use for parallel processing.
        """
        # Some healthy defensive programming.
        assert scalers is None or isinstance(scalers, dict)

        if model_path is not None:
            self._model = load_model(model_path)
        else:
            self._model: Optional[Model] = None

        self._history: Optional[pd.DataFrame] = None
        self._scalers: Optional[Dict[str, RobustScaler]] = scalers
        self._dag: Type[LayeredDAG] = dag
        self._adjacencies: Dict[str, np.ndarray] = self._dag.layer_adjacency_matrices()
        self._feature_settings: FeatureSettings = feature_settings
        self._verbose: bool = verbose
        self._n_jobs: Optional[int] = n_jobs
        self._metadata: Optional[Dict[str, Any]] = metadata
        self._features: List[Type[FeatureInterface]] = [
            feature_class(verbose=self._verbose, n_jobs=self._n_jobs)
            for feature_class in self._feature_settings.iter_features()
        ]

        feature_names = [
            feature.__class__.pythonic_name() for feature in self._features
        ]
        feature_class_names = [feature.__class__.__name__ for feature in self._features]
        assert len(set(feature_names)) == len(
            self._features
        ), f"Feature names not unique: {feature_names} ({feature_class_names})"

    @staticmethod
    def load_from_path(path: str) -> "Hammer":
        """Load a classifier model from a file."""

        if path.endswith(".tar.gz"):
            # We extract the files from the tar.gz file.
            os.system(f"tar -xzf {path}")

            path = path[:-7]

        dag_path = f"{path}/dag.pkl"
        scalers_path = f"{path}/scalers.pkl"
        model_path = f"{path}/model.keras"
        feature_settings_path = f"{path}/feature_settings.json"

        classifier = Hammer(
            dag=compress_pickle.load(dag_path),
            scalers=compress_pickle.load(scalers_path),
            feature_settings=FeatureSettings.from_json(feature_settings_path),
            model_path=model_path,
        )

        return classifier

    @staticmethod
    def load(model_name: str) -> "Hammer":
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

        return Hammer.load_from_path(path)

    def _build_input_modality(self, input_layer: Input) -> Layer:
        """Build the input modality sub-module."""
        hidden = input_layer

        if input_layer.shape[1] > 768:
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
            hidden = BatchNormalization(
                name=f"layer_normalization_{input_layer.name}_{i}"
            )(hidden)
        hidden = Dropout(0.3)(hidden)

        return hidden

    def _build_hidden_layers(self, inputs: List[Layer]) -> Layer:
        """Build the hidden layers sub-module."""
        if len(inputs) > 1:
            hidden = Concatenate(axis=-1)(inputs)
        else:
            hidden = inputs[0]
        for i in range(4):
            hidden = Dense(
                2048,
                activation="relu",
                kernel_initializer=HeNormal(),
                name=f"dense_hidden_{i}",
            )(hidden)
            hidden = BatchNormalization(
                name=f"layer_normalization_hidden_{i}",
            )(hidden)
        hidden = Dropout(0.3)(hidden)
        return hidden

    def _build_heads(self, hidden_output: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Build the output head sub-module."""
        outputs: Dict[str, tf.Tensor] = {}
        previous_output: Optional[tf.Tensor] = None
        for layer_name in self._dag.get_layer_names():
            if previous_output is None:
                previous_output = Dense(
                    self._dag.get_layer_size(layer_name),
                    name=layer_name,
                    activation="sigmoid",
                )(hidden_output)
                outputs[layer_name] = previous_output
                continue

            unharmonized_output: tf.Tensor = Dense(
                self._dag.get_layer_size(layer_name),
                name=f"unharmonized_{layer_name}",
                activation="sigmoid",
            )(hidden_output)

            harmonization: tf.Tensor = K.sum(
                previous_output[:, :, None] * self._adjacencies[layer_name][None, :, :],
                axis=1,
            )

            previous_output = Multiply(name=layer_name)(
                [unharmonized_output, harmonization]
            )

            outputs[layer_name] = previous_output

        return outputs

    def _build(self):
        """Build the classifier model."""
        input_layers: Dict[str, Input] = {
            feature.__class__.pythonic_name(): Input(
                shape=(feature.size(),),
                name=feature.__class__.pythonic_name(),
                dtype=feature.__class__.dtype(),
            )
            for feature in self._features
        }

        input_modalities: List[Layer] = [
            self._build_input_modality(input_layer)
            for input_layer in input_layers.values()
        ]

        hidden: Layer = self._build_hidden_layers(input_modalities)

        heads: Dict[str, tf.Tensor] = self._build_heads(hidden)

        self._model = Model(
            inputs=input_layers,
            outputs=heads,
            name="Hammer",
        )

    def _smiles_to_features(self, smiles: List[str]) -> Dict[str, np.ndarray]:
        """Transform a list of SMILES into a dictionary of features."""
        # First, we transform the SMILES to molecules.
        molecules = smiles_to_molecules(
            smiles, verbose=self._verbose, n_jobs=self._n_jobs
        )

        return {
            feature.__class__.pythonic_name(): feature.transform_molecules(molecules)
            for feature in tqdm(
                self._features,
                desc="Computing features",
                disable=not self._verbose,
                dynamic_ncols=True,
                leave=False,
            )
        }

    def fit(
        self,
        train_smiles: List[str],
        train_labels: Dict[str, np.ndarray],
        validation_smiles: List[str],
        validation_labels: Dict[str, np.ndarray],
        augmentation_settings: Optional[AugmentationSettings] = None,
        model_checkpoint_path: str = "checkpoints/model_checkpoint.keras",
        history_path: str = "histories/history.csv",
        history_plot_path: Optional[str] = None,
        model_plot_path: Optional[str] = None,
        maximal_number_of_epochs: int = 10_000,
        batch_size: int = 4096,
    ):
        """Train the classifier model."""
        self._build()
        self._model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics={
                layer_name: get_minimal_multiclass_metrics()
                for layer_name in self._dag.get_layer_names()
            },
        )

        if model_plot_path is not None:
            os.makedirs(os.path.dirname(model_plot_path), exist_ok=True)
            plot_model(
                self._model,
                to_file=model_plot_path,
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                expand_nested=True,
                dpi=100,
                show_layer_activations=True,
                show_trainable=True,
            )

        os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            model_checkpoint_path,
            monitor=f"val_{self._dag.leaf_layer_name}_AUPRC",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            verbose=0,
        )

        learning_rate_scheduler = ReduceLROnPlateau(
            monitor=f"val_{self._dag.leaf_layer_name}_AUPRC",
            factor=0.8,
            patience=20,
            verbose=0,
            mode="max",
            min_delta=1e-4,
            cooldown=5,
            min_lr=1e-6,
        )

        early_stopping = EarlyStopping(
            monitor=f"val_{self._dag.leaf_layer_name}_AUPRC",
            patience=100,
            verbose=0,
            mode="max",
            restore_best_weights=True,
        )

        if self._metadata is None:
            self._metadata = {}

        self._metadata["feature_settings"] = self._feature_settings.to_dict()
        self._metadata["batch_size"] = batch_size
        self._metadata["maximal_number_of_epochs"] = maximal_number_of_epochs

        train_smiles: List[str] = into_canonical_smiles(
            train_smiles, verbose=self._verbose, n_jobs=self._n_jobs
        )
        validation_smiles: List[str] = into_canonical_smiles(
            validation_smiles, verbose=self._verbose, n_jobs=self._n_jobs
        )

        if augmentation_settings is not None:
            (train_smiles, train_labels) = augmentation_settings.augment(
                train_smiles, train_labels, n_jobs=self._n_jobs, verbose=self._verbose
            )
            self._metadata["augmentation_settings"] = augmentation_settings.to_dict()
        else:
            self._metadata["augmentation_settings"] = None

        train_features: Dict[str, np.ndarray] = self._smiles_to_features(train_smiles)
        validation_features: Dict[str, np.ndarray] = self._smiles_to_features(
            validation_smiles
        )

        # We create a scaler for each feature that is not binary.
        self._scalers = {}
        for feature in self._features:
            if not feature.__class__.is_binary():
                self._scalers[feature.__class__.pythonic_name()] = RobustScaler()
                self._scalers[feature.__class__.pythonic_name()].fit(
                    train_features[feature.__class__.pythonic_name()]
                )
                train_features[feature.__class__.pythonic_name()] = self._scalers[
                    feature.__class__.pythonic_name()
                ].transform(train_features[feature.__class__.pythonic_name()])
                validation_features[feature.__class__.pythonic_name()] = self._scalers[
                    feature.__class__.pythonic_name()
                ].transform(validation_features[feature.__class__.pythonic_name()])

        training_history = self._model.fit(
            train_features,
            train_labels,
            epochs=maximal_number_of_epochs,
            callbacks=[
                TqdmCallback(
                    verbose=1 if self._verbose else 0,
                    metrics=[
                        "loss",
                        "val_loss",
                        *[
                            f"{prefix}{layer_name}_AUPRC"
                            for layer_name in self._dag.get_layer_names()
                            for prefix in ["", "val_"]
                        ],
                    ],
                ),
                model_checkpoint,
                TerminateOnNaN(),
                early_stopping,
                learning_rate_scheduler,
            ],
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
            validation_data=(validation_features, validation_labels),
        )

        self._history = pd.DataFrame(training_history.history)

        # We check if the path contains directories, and create them if they do not exist.
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        self._history.to_csv(history_path, index=False)

        if history_plot_path is not None:
            os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
            fig, _ = plot_history(
                self._history,
                monitor=f"val_{self._dag.leaf_layer_name}_AUPRC",
                monitor_mode="max",
            )
            fig.savefig(history_plot_path)
            plt.close(fig)

    def save(self, path: str):
        """Save the classifier model to a file."""
        assert self._model is not None, "Model not trained yet."

        tarball = path.endswith(".tar.gz")

        if path.endswith(".tar.gz"):
            path = path[:-7]

        # We create the directories if they do not exist.
        os.makedirs(path, exist_ok=True)

        dag_path = os.path.join(path, "dag.pkl")
        scalers_path = os.path.join(path, "scalers.pkl")
        model_path = os.path.join(path, "model.keras")

        compress_pickle.dump(self._dag, dag_path)
        compress_pickle.dump(self._scalers, scalers_path)
        self._model.save(model_path)

        # We compress the files into a single tar.gz file.
        if tarball:
            os.system(f"tar -czf {path}.tar.gz {path}")
            shutil.rmtree(path)

    def evaluate(
        self, smiles: List[str], labels: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate the classifier model."""
        assert self._model is not None, "Model not trained yet."

        smiles = into_canonical_smiles(
            smiles, verbose=self._verbose, n_jobs=self._n_jobs
        )
        features = self._smiles_to_features(smiles)

        for feature in self._features:
            if not feature.__class__.is_binary():
                features[feature.__class__.pythonic_name()] = self._scalers[
                    feature.__class__.pythonic_name()
                ].transform(features[feature.__class__.pythonic_name()])

        return self._model.evaluate(features, labels, verbose=0, return_dict=True)

    def predict_proba(self, smiles: List[str]) -> Dict[str, np.ndarray]:
        """Predict the probabilities of the classes."""
        assert self._model is not None, "Model not trained yet."

        smiles = into_canonical_smiles(
            smiles, verbose=self._verbose, n_jobs=self._n_jobs
        )
        features = self._smiles_to_features(smiles)

        for feature in self._features:
            if not feature.__class__.is_binary():
                features[feature.__class__.pythonic_name()] = self._scalers[
                    feature.__class__.pythonic_name()
                ].transform(features[feature.__class__.pythonic_name()])

        return self._model.predict(features)