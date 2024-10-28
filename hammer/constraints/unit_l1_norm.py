"""Submodule defining the UnitL1Norm keras constraint."""

from keras.api.constraints import Constraint  # type: ignore
from keras.api import Variable  # type: ignore
from keras.api import ops  # type: ignore
from keras.api.backend import epsilon  # type: ignore
from keras.api.saving import register_keras_serializable  # type: ignore


@register_keras_serializable(package="hammer")  # type: ignore
class UnitL1Norm(Constraint):
    """Unit L1 norm constraint for neural networks."""

    def __call__(self, w: Variable) -> Variable:
        """Apply the constraint to the weights."""
        return w / (epsilon() + ops.norm(w, ord=1))
