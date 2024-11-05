"""Submodule providing a custom transformer encoder layer."""

from typing import Union
from keras import ops  # type: ignore
from keras import KerasTensor
from keras.api.layers import Layer  # type: ignore
from keras.api.utils import register_keras_serializable  # type: ignore
from keras import activations
from keras import initializers
from keras.api.initializers import Initializer
from keras.api.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout


def clone_initializer(initializer: Union[Initializer, str]) -> Initializer:
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


def _check_masks_shapes(
    inputs: KerasTensor, padding_mask: KerasTensor, attention_mask: KerasTensor
) -> None:
    mask = padding_mask
    if hasattr(inputs, "_keras_mask") and mask is None:
        mask = inputs._keras_mask  # pylint: disable=protected-access
    if mask is not None:
        if len(mask.shape) != 2:
            raise ValueError(
                "`padding_mask` should have shape "
                "(batch_size, target_length). "
                f"Received shape `{mask.shape}`."
            )
    if attention_mask is not None:
        if len(attention_mask.shape) != 3:
            raise ValueError(
                "`attention_mask` should have shape "
                "(batch_size, target_length, source_length). "
                f"Received shape `{mask.shape}`."
            )


def merge_padding_and_attention_mask(
    inputs: KerasTensor,
    padding_mask: KerasTensor,
    attention_mask: KerasTensor,
) -> KerasTensor:
    """Merge the padding mask with a customized attention mask.

    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].

    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    _check_masks_shapes(inputs, padding_mask, attention_mask)
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask  # pylint: disable=protected-access
        else:
            raise ValueError(
                "If `inputs` has a `_keras_mask`, `padding_mask` should not be "
                "provided separately."
            )
    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        else:
            return ops.minimum(mask, attention_mask)
    return mask


@register_keras_serializable(package="hammer")
class TransformerEncoder(Layer):
    """Transformer encoder.

    This class follows the architecture of the transformer encoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up an encoder.

    This layer will correctly compute an attention mask from an implicit
    Keras padding mask (for example, by passing `mask_zero=True` to a
    `Embedding` layer). See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the
            `MultiHeadAttention` layer.
        dropout_rate: float. the dropout value, shared by
            `MultiHeadAttention` and feedforward network.
            Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer and the intermediate dense layer  are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        **kwargs: other keyword arguments passed to `Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:

    ```python
    # Create a single transformer encoder layer.
    encoder = keras_hub.layers.TransformerEncoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=(10, 64))
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = np.random.uniform(size=(2, 10, 64))
    output = model(input_data)
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout_rate=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True
        self._self_attention_layer = None
        self._self_attention_layer_norm = None
        self._self_attention_dropout = None
        self._feedforward_layer_norm = None
        self._feedforward_intermediate_dense = None
        self._feedforward_output_dense = None
        self._feedforward_dropout = None

    def build(self, input_shape):
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)
        if key_dim == 0:
            raise ValueError(
                "Attention `key_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(  # pylint: disable=protected-access
                query=input_shape,
                value=input_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=input_shape,
                value_shape=input_shape,
            )
        self._self_attention_layer_norm = LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(input_shape)
        self._self_attention_dropout = Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_layer_norm = LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(input_shape)
        self._feedforward_intermediate_dense = Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(input_shape)
        self._feedforward_output_dense = Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    # pylint: disable=arguments-differ
    def call(
        self,
        inputs,
        padding_mask=None,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ) -> KerasTensor:
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, hidden_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].
            training: a boolean indicating whether the layer should behave in
                training mode or in inference mode.
            return_attention_scores: a boolean indicating whether the output
                should be `(attention_output, attention_scores)` if `True` or
                `attention_output` if `False`. Defaults to `False`.

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        x = inputs  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layer_norm(x)

        if return_attention_scores:
            x, attention_scores = self._self_attention_layer(
                query=x,
                value=x,
                attention_mask=self_attention_mask,
                return_attention_scores=return_attention_scores,
                training=training,
            )
            return x, attention_scores
        else:
            x = self._self_attention_layer(
                query=x,
                value=x,
                attention_mask=self_attention_mask,
                training=training,
            )

        x = self._self_attention_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layer_norm(x)

        if return_attention_scores:
            return x, attention_scores

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "activation": activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "normalize_first": self.normalize_first,
            }
        )
        return config

    # pylint: disable=arguments-differ
    def compute_output_shape(self, input_shape):
        """Return the input shape."""
        return input_shape

    def _serialize_to_tensors(self) -> dict:
        """Handle serialization to tensors (needed for checkpointing)."""
        return {}

    def _restore_from_tensors(self, restored_tensors: dict) -> None:
        """Handle restoration from tensors (needed for checkpointing)."""

    def add_metric(self, *args, **kwargs) -> None:
        """Add a metric to the layer (not used in this layer)."""
        # This is a placeholder since we are not adding any custom metrics in this layer
