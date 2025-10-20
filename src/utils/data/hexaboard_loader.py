import json
from typing import Set, Tuple

import numpy as np


def load_hexaboard(path: str, normalize: bool = True) -> np.ndarray:
    """
    Loads a hexaboard from a .npy file, converts color channels from BGR to RGB,
    and optionally normalizes pixel values to [0, 1].

    Parameters
    ----------
    path: str
        Path to the .npy file containing the hexaboard data.
    normalize: bool, optional
        If True, normalize pixel values to the [0, 1] range.
    
    Returns
    -------
    np.ndarray
        The processed hexaboard as a 5D numpy array (H_seg, V_seg, height, width, num_channels).
    """
    hexaboard = np.load(path)
    hexaboard = hexaboard[..., ::-1].copy()  # convert from BGR to RGB

    # Normalize pixel values to [0, 1] range if specified
    if normalize:
        hexaboard = (hexaboard / 255.0).astype(np.float32)

    assert hexaboard.ndim == 5, "Hexaboard array must be 5D (H_seg, V_seg, height, width, num_channels)."

    return hexaboard


def load_skipped_segments(path: str) -> Set[Tuple[int, int]]:
    """
    Loads a JSON file containing indices of hexaboard segments to skip during processing.

    Parameters
    ----------
    path: str
        Path to the JSON file containing the list of segments to skip.

    Returns
    -------
    Set[Tuple[int, int]]
        A set of tuples representing the (H_seg, V_seg) indices of segments to skip.
    """
    with open(path, 'r') as f:
        skipped_segments = json.load(f)
        
    return set(map(tuple, skipped_segments))