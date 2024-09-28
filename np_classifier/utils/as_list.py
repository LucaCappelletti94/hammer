"""Normalizes the provided index to a list of strings."""

from typing import List, Dict


def as_list(index: Dict[str, int]) -> List[str]:
    """Normalizes the provided index to a list of strings.

    Parameters
    ----------
    index : Dict[str, int]
        The index to normalize.

    """
    reversed_index: Dict[int, str] = {v: k for k, v in index.items()}
    return [reversed_index[i] for i in range(len(reversed_index))]
