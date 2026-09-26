import os
from typing import Tuple

import numpy as np
import pytest

import torch

from src.inspection import autoencoder_inference, pixelwise_inference
from src.utils import set_seed
from src.utils.data import load_hexaboard

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = './data/train'
LOG_DIR = './logs'


@pytest.fixture
def config() -> Tuple:
    input_channels = 3
    height = 1060
    width = 1882
    latent_dim = 32
    init_filters = 128
    layers = [2, 2, 2]

    return input_channels, height, width, latent_dim, init_filters, layers


@pytest.fixture
def baseline_hexaboard() -> np.ndarray:
    return load_hexaboard(os.path.join(DATA_DIR, '320XLF4CQH00001.npy'))


@pytest.fixture
def temp_hexaboard() -> np.ndarray:
    # Create a temporary 5D array with random data
    temp_data = np.random.rand(8, 5, 100, 100, 3)  # smaller size for faster tests

    return temp_data


def test_identical_pixelwise_inference(temp_hexaboard: np.ndarray) -> None:
    H_seg, V_seg, _, _, _ = temp_hexaboard.shape
    threshold = np.full((H_seg, V_seg), 0.9, dtype=float)  # 2D threshold array
    flagged_segments = pixelwise_inference(temp_hexaboard, temp_hexaboard, threshold)

    assert len(flagged_segments) == 0, "No segments should be flagged for identical images."


def test_one_diff_pixelwise_inference(temp_hexaboard: np.ndarray) -> None:
    # Create a modified version
    new_hexaboard = temp_hexaboard.copy()
    new_hexaboard[0, 0, :, :, :] = new_hexaboard[0, 0, :, :, :] * 0.5

    H_seg, V_seg, _, _, _ = temp_hexaboard.shape
    threshold = np.full((H_seg, V_seg), 0.1, dtype=float)

    flagged_segments = pixelwise_inference(temp_hexaboard, new_hexaboard, threshold, skipped_segments=set())
    assert len(flagged_segments) >= 1, "At least one segment should be flagged for different images."
    assert (0, 0) in flagged_segments, "The modified segment should be flagged."


def test_identical_autoencoder_inference(temp_hexaboard: np.ndarray) -> None:
    board = temp_hexaboard.astype(np.float32)
    threshold = np.ones(board.shape[:2])
    flags = autoencoder_inference(board, threshold, torch.nn.Identity(), torch.device('cpu'))
    assert flags == []


def test_pixelwise_inference_shape_mismatch() -> None:
    baseline = np.random.rand(8, 5, 100, 100, 3).astype(np.uint8)
    new = np.random.rand(7, 5, 100, 100, 3).astype(np.uint8)  # different H_seg
    threshold = np.zeros((8, 5), dtype=float)  # 2D threshold array

    with pytest.raises(ValueError, match="Baseline and new hexaboards must have the same shape."):
        pixelwise_inference(baseline, new, threshold)


def test_autoencoder_inference_wrong_dims() -> None:
    with pytest.raises(ValueError, match="Expected 5D array"):
        autoencoder_inference(
            np.zeros((10, 10, 3), dtype=np.float32),
            np.zeros((8, 5)),
            torch.nn.Identity(),
            torch.device('cpu')
        )
