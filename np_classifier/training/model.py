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
from tqdm.keras import TqdmCallback
import numpy as np
import pandas as pd
from plot_keras_history import plot_history


class Classifier:
    """Class representing the multi-modal multi-class classifier model."""

    def __init__(self):
        """Initialize the classifier model."""
        self._model: Optional[Model] = None
        self._history: Optional[pd.DataFrame] = None

    def _build_input_modality(self, input_layer: Input) -> Layer:
        """Build the input modality sub-module."""
        hidden = Dense(512, activation="relu", name=f"dense_{input_layer.name}")(
            input_layer
        )
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.5)(hidden)
        return hidden

    def _build_hidden_layers(self, inputs: List[Layer]) -> Layer:
        """Build the hidden layers sub-module."""
        hidden = Concatenate(axis=-1)(inputs)
        hidden = Dense(1024, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(512, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(1024, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(512, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.5)(hidden)
        return hidden

    def _build_output_head(
        self, name: str, input_layer: Layer, output_size: int
    ) -> (Layer, Layer):
        """Build the output head sub-module."""
        hidden = Dense(512, activation="relu")(input_layer)
        output = Dense(output_size, name=name, activation="sigmoid")(hidden)
        return (hidden, output)

    def _build(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]):
        """Build the classifier model."""
        input_layers: List[Input] = [
            Input(shape=(input_tensor.shape[1],), name=name)
            for name, input_tensor in inputs.items()
        ]

        input_modalities: List[Layer] = [
            self._build_input_modality(input_layer) for input_layer in input_layers
        ]

        hidden: Layer = self._build_hidden_layers(input_modalities)

        pathway_hidden, pathway_head = self._build_output_head(
            "pathway_head", hidden, outputs["pathway"].shape[1]
        )
        concatenated_pathway = Concatenate(
            axis=-1,
            name="concatenated_pathway",
        )([hidden, pathway_hidden])
        superclass_hidden, superclass_head = self._build_output_head(
            "superclass_head", concatenated_pathway, outputs["superclass"].shape[1]
        )
        concatenated_pathway_and_superclass = Concatenate(
            axis=-1,
            name="concatenated_pathway_and_superclass",
        )([concatenated_pathway, superclass_hidden])
        _, class_head = self._build_output_head(
            "class_head", concatenated_pathway_and_superclass, outputs["class"].shape[1]
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
    ):
        """Train the classifier model."""
        self._build(*train)
        self._model.compile(optimizer="adam", loss="binary_crossentropy")
        self._model.summary()
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

        training_history = self._model.fit(
            *train,
            epochs=10_000,
            callbacks=[TqdmCallback(verbose=2)],
            batch_size=2048,
            shuffle=True,
            verbose=0,
            validation_data=val
        )
        self._history = pd.DataFrame(training_history.history)

        self._history.to_csv("history.csv")

    def plot_training_history(self):
        """Plot the training history."""
        if self._history is None:
            raise ValueError("No training history available.")
        plot_history(self._history)

    def evaluate(self, test: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]):
        """Evaluate the classifier model."""
        return self._model.evaluate(test[0].values(), test[1].values())
