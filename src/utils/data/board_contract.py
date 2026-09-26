from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BOARD_SHAPE = (8, 5, 1060, 1882, 3)
GRID_SHAPE = BOARD_SHAPE[:2]
SKIPPED_SEGMENTS_PATH = REPO_ROOT / 'calibrations' / 'skipped_segments.json'


def validate_board(array: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = BOARD_SHAPE) -> None:
    if array.ndim != 5 or array.shape[-1] != 3 or any(size <= 0 for size in array.shape):
        raise ValueError(f"Expected board shape `[rows, cols, height, width, 3]`, got `{array.shape}`.")

    if expected_shape is not None and array.shape != tuple(expected_shape):
        raise ValueError(f"Expected board shape `{expected_shape}`, got `{array.shape}`.")

    if array.dtype != np.uint8:
        raise ValueError(f"Stored boards must use `uint8` BGR pixels, got `{array.dtype}`.")


def validate_coordinates(coordinates: Iterable[Tuple[int, int]], shape: Tuple[int, int]) -> Set[Tuple[int, int]]:
    result = set()
    for coordinate in coordinates:
        if len(coordinate) != 2:
            raise ValueError(f"Invalid segment coordinate: `{coordinate}`.")

        row, col = coordinate
        if (
            isinstance(row, bool) or isinstance(col, bool)
            or not isinstance(row, (int, np.integer))
            or not isinstance(col, (int, np.integer))
            or not (0 <= row < shape[0] and 0 <= col < shape[1])
        ):
            raise ValueError(f"Segment coordinate `{coordinate}` is outside grid `{shape}` or is not integral.")

        result.add((int(row), int(col)))

    return result


def open_board(path: str, expected_shape: Optional[Tuple[int, ...]] = BOARD_SHAPE) -> np.ndarray:
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    validate_board(array, expected_shape)
    return array


def rgb_segment(board: np.ndarray, row: int, col: int, normalize: bool = True) -> np.ndarray:
    segment = np.ascontiguousarray(board[row, col, ..., ::-1])
    if normalize:
        return segment.astype(np.float32) / 255.0

    return segment
