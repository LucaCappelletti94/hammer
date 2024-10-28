"""Submodule providing an harmonization graph convolution layer."""

from typing import List, Tuple, Type, Optional, Union, Callable
from keras import ops  # type: ignore
from keras.api.initializers import Initializer  # type: ignore
from keras.api import initializers  # type: ignore
from keras.api import activations  # type: ignore
from keras.api.regularizers import Regularizer  # type: ignore
from keras.api import regularizers  # type: ignore
from keras.api.constraints import Constraint  # type: ignore
from keras.api import constraints  # type: ignore
from keras.api.layers import Layer  # type: ignore
from keras.api.utils import register_keras_serializable  # type: ignore
from keras.api import Variable  # type: ignore
from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf  # type: ignore


# pylint: disable=too-many-ancestors
@register_keras_serializable(package="hammer")
class HarmonizeGraphConvolution(Layer):
    """Harmonization graph convolution layer for neural networks."""

    def __init__(
        self,
        supports: List[csr_matrix],
        activation: Optional[str] = None,
        kernel_initializer: Union[str, Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[Regularizer] = None,
        activity_regularizer: Optional[Regularizer] = None,
        kernel_constraint: Optional[Constraint] = None,
        **kwargs,
    ) -> None:
        """Initialize the layer with the adjacency matrix.

        Parameters
        ----------
        supports : List[csr_matrix]
            The adjacency matrix that defines the valid connections between the parents
            and children. It is expected to be a 2D numpy array with shape (n_parents,
            n_children).
        activation : str, optional
            The activation function to use, by default None
        kernel_initializer : Union[str, Initializer], optional
            The initializer for the kernel weights, by default 'glorot_uniform'
        kernel_regularizer : Optional[Regularizer], optional
            The regularizer for the kernel weights, by default None
        activity_regularizer : Optional[Regularizer], optional
            The regularizer for the activity, by default None
        kernel_constraint : Optional[Constraint], optional
            The constraint for the kernel weights, by default None
        """
        super().__init__(**kwargs)

        # It must have at least one support
        assert len(supports) > 0, "At least one support must be provided"

        # The number of nodes in all supports must be the same
        assert (
            len(set(support.shape[0] for support in supports)) == 1
        ), "The number of nodes in all supports must be the same"

        self.supports: List[csr_matrix] = supports
        self.activation: Type[Callable] = activations.get(activation)
        self.kernel_initializer: Union[str, Initializer] = kernel_initializer
        self.kernel_regularizer: Optional[Regularizer] = kernel_regularizer
        self.activity_regularizer: Optional[Regularizer] = activity_regularizer
        self.kernel_constraint: Optional[Constraint] = kernel_constraint
        # List of kernel values, one for each support, large as the number of
        # edges in each support
        self.kernels: List[Variable] = []
        # Bias term
        self.bias: Variable = None

    # pylint: disable=arguments-differ
    def compute_output_shape(self, input_shapes) -> Tuple[int]:
        """Compute the output shape of the layer."""
        return input_shapes

    # pylint: disable=arguments-renamed
    def build(self, input_shapes) -> None:
        """Build the layer with the input shapes."""
        self.kernels = [
            self.add_weight(
                shape=(support.nnz,),
                initializer=self.kernel_initializer,
                name=f"kernel_{i}",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            for i, support in enumerate(self.supports)
        ]
        self.bias = self.add_weight(
            shape=(self.supports[0].shape[0],),
            initializer="zeros",
            name="bias",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.built = True

    # pylint: disable=arguments-differ
    def call(self, features):
        supports: List[tf.Tensor] = []  # type: ignore
        for support, kernel in zip(self.supports, self.kernels):

            sparse_weights = tf.SparseTensor(
                indices=np.column_stack((support.nonzero())).tolist(),
                values=kernel * support.data.astype(np.float32),
                dense_shape=support.shape,
            )

            supports.append(
                tf.transpose(tf.sparse.sparse_dense_matmul(
                    sparse_weights,
                    tf.transpose(features),
                ))
            )

        # Then we sum all the supports and apply the activation function
        return self.activation(tf.add_n(supports) + self.bias)

    @classmethod
    def from_config(cls, config: dict) -> "HarmonizeGraphConvolution":
        """Create a layer from the configuration."""
        supports = [
            csr_matrix(
                (support["data"], support["indices"], support["indptr"]),
                shape=support["shape"],
            )
            for support in config["supports"]
        ]
        config["supports"] = supports
        return cls(**config)

    def get_config(self):
        """Return the configuration of the layer."""
        return {
            "supports": [
                {
                    "indices": support.indices.tolist(),
                    "indptr": support.indptr.tolist(),
                    "data": support.data.tolist(),
                    "shape": support.shape,
                }
                for support in self.supports
            ],
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            **super().get_config(),
        }

    def _serialize_to_tensors(self) -> dict:
        """Handle serialization to tensors (needed for checkpointing)."""
        return {}

    def _restore_from_tensors(self, restored_tensors: dict) -> None:
        """Handle restoration from tensors (needed for checkpointing)."""

    def add_metric(self, *args, **kwargs) -> None:
        """Add a metric to the layer (not used in this layer)."""
        # This is a placeholder since we are not adding any custom metrics in this layer
