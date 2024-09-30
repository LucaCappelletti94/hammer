"""Submodule providing utility to convert a list of indices to a one-hot numpy array."""

from typing import List
import numpy as np


def as_one_hot(indices: List[int], size: int) -> np.ndarray:
    """Converts a list of indices to a one-hot numpy array.

    Parameters
    ----------
    indices : List[int]
        The list of indices to convert.
    size : int
        The size of the one-hot array.

    Returns
    -------
    np.ndarray
        The one-hot numpy array.

    """
    one_hot = np.zeros(size)
    one_hot[indices] = 1
    return one_hot
