"""Submodule providing an harmonization layer for neural networks."""

from typing import Tuple
from keras import ops
from keras import KerasTensor
from keras.api.layers import Layer
import numpy as np


# pylint: disable=too-many-ancestors
class Harmonize(Layer):
    """Harmonization layer for neural networks.

    This layer is used to capture the hierarchical structure of the output DAG. It
    receives the output of two layers, the parent layer and the child layer, both of
    which are expected to have Sigmoid activations. Considering the shape of the first
    layer to be (batch_size, n_parents, n_features) and the shape of the second layer
    to be (batch_size, n_children, n_features), plus an adjacency matrix that is
    defined in the constructor of the layer itself which represents the set of known
    valid connections between the parents and children, the layer will output a tensor
    of shape (batch_size, n_children, n_features) where each element of the output
    tensor is a weighted sum of the parents of the corresponding child, with the weights
    being determined by the adjacency matrix.

    This has the goal to harmonize the information of the parents of a child in a way
    that the child predictions will always necessarily comply with the DAG structure.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        The adjacency matrix that defines the valid connections between the parents
        and children. It is expected to be a 2D numpy array with shape (n_parents,
        n_children).
    """

    def __init__(self, adjacency_matrix: np.ndarray, **kwargs) -> None:
        """Initialize the layer with the adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            The adjacency matrix that defines the valid connections between the parents
            and children. It is expected to be a 2D numpy array with shape (n_parents,
            n_children).
        """
        super().__init__(**kwargs)
        assert isinstance(adjacency_matrix, np.ndarray)
        assert adjacency_matrix.ndim == 2
        assert adjacency_matrix.shape[0] > 0
        assert adjacency_matrix.shape[1] > 0
        self.adjacency_matrix: np.ndarray = adjacency_matrix

    def call(self, *args, **kwargs) -> KerasTensor:
        """Perform the forward pass of the layer.

        Parameters
        ----------
        inputs : Tuple[KerasTensor, KerasTensor]
            A tuple containing two tensors, the first one representing the parents and
            the second one representing the children. Both tensors are expected to have
            the shape (batch_size, n_nodes, n_features).

        Returns
        -------
        KerasTensor
            The output tensor of the layer, with shape (batch_size, n_children,
            n_features).
        """
        parents, children = args
        assert parents.shape[1] == self.adjacency_matrix.shape[1], (
            f"Harmonize({self.name}): Expected the number "
            f"of parents to be {self.adjacency_matrix.shape[1]}, "
            f"but got {parents.shape[1]}. "
            f"Adjacency matrix shape: {self.adjacency_matrix.shape}."
        )
        assert children.shape[1] == self.adjacency_matrix.shape[0], (
            f"Harmonize({self.name}): Expected the number "
            f"of children to be {self.adjacency_matrix.shape[0]}, "
            f"but got {children.shape[1]}. "
            f"Adjacency matrix shape: {self.adjacency_matrix.shape}."
        )

        # Compute the harmonization weights
        harmonization_weights: KerasTensor = ops.dot(parents, self.adjacency_matrix.T)

        if harmonization_weights.shape != children.shape:
            raise ValueError(
                f"Harmonize({self.name}): Expected the output shape to be {children.shape}, "
                f"but got {harmonization_weights.shape}."
            )

        # Since the task is a multi-label classification, we cannot
        # use the softmax activation function, as it would force the
        # sum of the weights to be 1. Instead, we use the Sigmoid activation
        # after the batch_dot product to ensure that the weights are in the
        # interval [0, 1].

        # Compute the output tensor
        output: KerasTensor = harmonization_weights * children

        if output.shape != children.shape:
            raise ValueError(
                f"Harmonize({self.name}): Expected the output shape to be {children.shape}, "
                f"but got {harmonization_weights.shape}."
            )

        return output

    # pylint: disable=arguments-differ
    def compute_output_shape(self) -> Tuple:
        """Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : Tuple
            The shape of the input tensors, represented as a tuple of tuples.

        Returns
        -------
        Tuple
            The shape of the output tensor, represented as a tuple.
        """
        return (
            None,
            self.adjacency_matrix.shape[0],
        )

    def get_config(self) -> dict:
        """Return the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "adjacency_matrix": self.adjacency_matrix.tolist(),
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "Harmonize":
        """Create a layer from the configuration."""
        return cls(
            adjacency_matrix=np.array(config["adjacency_matrix"]),
            name=config["name"],
        )

    def _serialize_to_tensors(self) -> dict:
        """Handle serialization to tensors (needed for checkpointing)."""
        return {}

    def _restore_from_tensors(self, restored_tensors: dict) -> None:
        """Handle restoration from tensors (needed for checkpointing)."""

    def add_metric(self, *args, **kwargs) -> None:
        """Add a metric to the layer (not used in this layer)."""
        # This is a placeholder since we are not adding any custom metrics in this layer
