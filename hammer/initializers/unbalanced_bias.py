"""Bias inizialer for unbalanced prediction tasks."""

from keras.api.initializers import Initializer
from keras.api.backend import epsilon
from keras.api.utils import register_keras_serializable
import numpy as np


@register_keras_serializable(package="hammer")
class LogitBiasInitializer(Initializer):
    """Initializer that sets the bias based on the prior distribution of the training labels."""

    def __init__(self, prior_probs: np.ndarray):
        """
        Create a new initializer with the prior probabilities of the labels.

        :param prior_probs: numpy array of prior probabilities for each label
        """
        self.prior_probs = prior_probs

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "LogitBiasInitializer":
        """
        Create an initializer from the one-hot encoded labels.

        :param labels: numpy array of one-hot encoded labels
        :return: the initializer object
        """
        # We check that labels is a numpy array
        if not isinstance(labels, np.ndarray):
            raise ValueError("Labels must be a numpy array.")

        # We check that labels is a 2D array
        if len(labels.shape) != 2:
            raise ValueError("Labels must be a 2D array.")

        # We check that labels is a binary array
        if not np.array_equal(np.unique(labels), np.array([0, 1])):
            raise ValueError("Labels must be a binary array.")

        # Calculate the prior probability for each label
        label_sums = np.sum(labels, axis=0)
        total_samples = labels.shape[0]
        prior_probs = label_sums / total_samples

        return cls(prior_probs)

    def __call__(self, shape, dtype=None):
        """
        Compute the bias values based on the log-odds of each class.

        :param shape: shape of the bias tensor (should match the number of output classes)
        :param dtype: the dtype of the tensor (optional)
        :return: initialized bias tensor
        """
        # Prevent divide by zero or log of zero issues
        log_odds = np.log(
            epsilon() + self.prior_probs / (1 - self.prior_probs + epsilon())
        )

        # In some unfortunate cases, such as when no sample has a certain label,
        # the log-odds will have as value -inf. We replace these values with 0.
        log_odds[np.isneginf(log_odds)] = 0

        # We check that there is no NaN in the log-odds
        if np.isnan(log_odds).any():
            raise ValueError("NaN values found in the log-odds.")

        # Ensure the shape matches the expected number of output classes
        if len(log_odds) != shape[0]:
            raise ValueError(
                f"Shape mismatch: expected {shape[0]} biases, but found {len(log_odds)} log-odds."
            )

        # Return the log-odds as the bias initialization
        return log_odds.astype(dtype)

    @classmethod
    def from_config(cls, config: dict) -> "LogitBiasInitializer":
        """
        Create an initializer from a configuration dictionary.

        :param config: configuration dictionary
        :return: the initializer object
        """
        return cls(prior_probs=np.array(config["prior_probs"]))

    def get_config(self):
        """
        Return the configuration of the initializer for serialization.
        """
        return {"prior_probs": self.prior_probs.tolist()}
