import json
from typing import Optional, Set, Tuple

import numpy as np

from .board_contract import BOARD_SHAPE, GRID_SHAPE, SKIPPED_SEGMENTS_PATH, open_board, validate_coordinates


def load_hexaboard(
    path: str,
    normalize: bool = True,
    expected_shape: Optional[Tuple[int, ...]] = BOARD_SHAPE
) -> np.ndarray:
    """
    Load a stored BGR board as RGB, optionally scaled to `[0, 1]`.

    Parameters
    ----------
    path : str
        NumPy board path.
    normalize : bool, optional
        Return float32 pixels instead of uint8 pixels.
    expected_shape : Tuple[int, ...], optional
        Required production shape; use an explicit smaller shape for fixtures.

    Returns
    -------
    np.ndarray
        RGB board with shape `[rows, cols, height, width, 3]`.
    """
    board = open_board(path, expected_shape)
    rgb = np.ascontiguousarray(board[..., ::-1])
    if normalize:
        return rgb.astype(np.float32) / 255.0

    return rgb


def load_skipped_segments(
    path: Optional[str] = None,
    shape: Tuple[int, int] = GRID_SHAPE
) -> Set[Tuple[int, int]]:
    """
    Read and validate the canonical skip mask.

    Parameters
    ----------
    path : str, optional
        JSON path; defaults to the repository calibration mask.
    shape : Tuple[int, int], optional
        Grid bounds.

    Returns
    -------
    Set[Tuple[int, int]]
        Zero-based skipped coordinates.
    """
    with open(path or SKIPPED_SEGMENTS_PATH, 'r', encoding='utf-8') as handle:
        coordinates = json.load(handle)

    return validate_coordinates(coordinates, shape)
