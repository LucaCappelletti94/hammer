"""Submodule providing the multi-modal multi-class classifier model.

Implementative details
----------------------
The model is a feed-forward neural network based on Keras/TensorFlow.

Multimodality
~~~~~~~~~~~~~~~~~~~~~~
The model receives a dictionary of inputs, where the keys are the name
of the modality and the values are the input tensors.
For each modality, it initializes a separate sub-module. The sub-modules
are responsible for processing the input tensors of the corresponding modality.

Multiclass
~~~~~~~~~~~~~~~~~~~~~~
The model receives three main output tensors, one for each class level
(pathway, superclass, class). Each output tensor is itself a vector of
binary values, where each value corresponds to a class label.
Some samples may have multiple class labels for any given class level,
and as such, the model uses a binary cross-entropy loss function for
each output tensor, with a sigmoid activation function. Each head of
the model is a separate sub-module, responsible for processing the output
tensor of the corresponding class level.

Principal layers
~~~~~~~~~~~~~~~~~~~~~~
The model uses a combination of dense layers, batch normalization layers,
and dropout layers. The dense layers are responsible for the main processing
of the input tensors, while the batch normalization layers are used to
normalize the activations of the previous layer at each batch. The dropout
layers are used to prevent overfitting by randomly setting a fraction of
the input units to 0 at each update during training. By default, the dropout
rate is set to 0.5. The model also uses ReLU activation functions for the
dense layers. As aforementioned, the output layers use a sigmoid activation
function.

Training strategy
~~~~~~~~~~~~~~~~~~~~~~
The model optimizer is Adam, with the default learning rate of 0.001.
The model also employs early stopping with a patience of 100 epochs,
with a maximal number of epochs set to 10_000, based on the training
loss. 


"""

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
    Dropout,  # pylint: disable=no-name-in-module,import-error
    BatchNormalization,  # pylint: disable=no-name-in-module,import-error
)
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
from tqdm.keras import TqdmCallback
import numpy as np
import pandas as pd
from plot_keras_history import plot_history
from extra_keras_metrics import get_standard_binary_metrics


class Classifier:
    """Class representing the multi-modal multi-class classifier model."""

    def __init__(self, number_of_epochs: int = 10_000):
        """Initialize the classifier model."""
        self._model: Optional[Model] = None
        self._history: Optional[pd.DataFrame] = None
        self._number_of_epochs = number_of_epochs

    def _build_input_modality(self, input_layer: Input) -> Layer:
        """Build the input modality sub-module."""
        hidden = input_layer

        if input_layer.shape[1] == 2048:
            hidden_sizes = 768
        else:
            hidden_sizes = 128

        for i in range(5):
            hidden = Dense(
                hidden_sizes,
                activation="relu",
                kernel_initializer=HeNormal(),
                name=f"dense_{input_layer.name}_{i}",
            )(hidden)
            hidden = BatchNormalization(
                name=f"batch_normalization_{input_layer.name}_{i}"
            )(hidden)
            hidden = Dropout(
                0.4,
                name=f"dropout_{input_layer.name}_{i}",
            )(hidden)
        return hidden

    def _build_hidden_layers(self, inputs: List[Layer]) -> Layer:
        """Build the hidden layers sub-module."""
        hidden = Concatenate(axis=-1)(inputs)
        for i in range(10):
            hidden = Dense(
                2048,
                activation="relu",
                kernel_initializer=HeNormal(),
                name=f"dense_hidden_{i}",
            )(hidden)
            hidden = BatchNormalization(
                name=f"batch_normalization_hidden_{i}",
            )(hidden)
            hidden = Dropout(
                0.6,
                name=f"dropout_hidden_{i}",
            )(hidden)
        return hidden

    def _build_pathway_head(self, input_layer: Layer, number_of_pathways: int) -> Layer:
        """Build the output head sub-module."""
        return Dense(number_of_pathways, name="pathway", activation="sigmoid")(
            input_layer
        )

    def _build_superclass_head(
        self, input_layer: Layer, number_of_superclasses: int
    ) -> Layer:
        """Build the output head sub-module."""
        return Dense(number_of_superclasses, name="superclass", activation="sigmoid")(
            input_layer
        )

    def _build_class_head(self, input_layer: Layer, number_of_classes: int) -> Layer:
        """Build the output head sub-module."""
        return Dense(number_of_classes, name="class", activation="sigmoid")(input_layer)

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

        input_layers: List[Input] = [
            Input(shape=input_array.shape[1:], name=name, dtype=input_array.dtype)
            for name, input_array in inputs.items()
        ]

        input_modalities: List[Layer] = [
            self._build_input_modality(input_layer) for input_layer in input_layers
        ]

        hidden: Layer = self._build_hidden_layers(input_modalities)

        pathway_head = self._build_pathway_head(hidden, outputs["pathway"].shape[1])
        superclass_head = self._build_superclass_head(
            hidden, outputs["superclass"].shape[1]
        )
        class_head = self._build_class_head(hidden, outputs["class"].shape[1])

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
    ):
        """Train the classifier model."""
        self._build(*train)
        self._model.compile(
            optimizer=Adam(clipnorm=1.0),
            loss="binary_focal_crossentropy",
            metrics={
                "pathway": get_standard_binary_metrics(),
                "superclass": get_standard_binary_metrics(),
                "class": get_standard_binary_metrics(),
            },
        )
        plot_model(
            self._model,
            to_file="model.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=200,
            show_layer_activations=True,
            show_trainable=True,
        )

        if holdout_number is not None:
            model_checkpoint_path = f"model_checkpoint_{holdout_number}.keras"
        else:
            model_checkpoint_path = "model_checkpoint.keras"

        model_checkpoint = ModelCheckpoint(
            model_checkpoint_path,
            monitor="val_class_mcc",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            verbose=0,
        )

        learning_rate_scheduler = ReduceLROnPlateau(
            monitor="val_class_mcc",  # Monitor the validation loss to avoid overfitting.
            factor=0.8,  # Reduce the learning rate by a small factor (e.g., 20%) to prevent abrupt drops.
            patience=100,  # Wait for 20 epochs without improvement before reducing LR (long patience to allow grokking).
            verbose=1,  # Verbose output for logging learning rate reductions.
            mode="max",  # Minimize the validation loss.
            min_delta=1e-4,  # Small change threshold for improvement, encouraging gradual learning.
            cooldown=150,  # After a learning rate reduction, wait 10 epochs before resuming normal operation.
            min_lr=1e-6,  # Set a minimum learning rate to avoid reducing it too much and stalling learning.
        )

        early_stopping = EarlyStopping(
            monitor="val_class_mcc",
            patience=1000,
            verbose=1,
            mode="max",
            restore_best_weights=True,
        )

        training_history = self._model.fit(
            *train,
            epochs=self._number_of_epochs,
            callbacks=[
                TqdmCallback(
                    verbose=1,
                    metrics=[
                        "loss",
                        "val_loss",
                        "class_mcc",
                        "val_class_mcc",
                        "superclass_mcc",
                        "val_superclass_mcc",
                        "pathway_mcc",
                        "val_pathway_mcc",
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
            self._history, monitor="val_class_mcc", monitor_mode="max"
        )

        # We create a directory 'histories' if it does not exist.
        os.makedirs("histories", exist_ok=True)

        if holdout_number is not None:
            self._history.to_csv(f"histories/history_{holdout_number}.csv")
            fig.savefig(f"histories/history_{holdout_number}.png")
        else:
            self._history.to_csv("histories/history.csv")
            fig.savefig("histories/history.png")

    def evaluate(
        self, test: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Evaluate the classifier model."""
        return self._model.evaluate(*test, verbose=0, return_dict=True)
