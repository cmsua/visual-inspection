import os
from typing import Tuple

import pytest
import numpy as np

import torch

from src.inferences import autoencoder_inference, pixelwise_inference
from src.models import CNNAutoencoder
from src.utils import set_seed
from src.utils.data import load_hexaboard

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = './data/train'
LOG_DIR = './logs'


@pytest.fixture
def config() -> Tuple:
    input_channels = 3
    height = 1016
    width = 1640
    latent_dim = 32
    init_filters = 128
    layers = [2, 2, 2]

    return input_channels, height, width, latent_dim, init_filters, layers


@pytest.fixture
def baseline_hexaboard() -> np.ndarray:
    return load_hexaboard(os.path.join(DATA_DIR, 'aligned_images1.npy'))


@pytest.fixture
def temp_hexaboard() -> np.ndarray:
    # Create a temporary 5D array with random data
    temp_data = np.random.rand(13, 9, 100, 100, 3)  # smaller size for faster tests

    return temp_data


def test_identical_pixelwise_inference(temp_hexaboard: np.ndarray) -> None:
    H_seg, V_seg, _, _, _ = temp_hexaboard.shape
    threshold = np.full((H_seg, V_seg), 0.9, dtype=float)  # 2D threshold array
    flagged_segments = pixelwise_inference(temp_hexaboard, temp_hexaboard, threshold)

    assert len(flagged_segments) == 0, "No segments should be flagged for identical images."


def test_one_diff_pixelwise_inference(temp_hexaboard: np.ndarray) -> None:
    # Create a modified version
    new_hexaboard = temp_hexaboard.copy()
    new_hexaboard[0, 0, :, :, :] = (new_hexaboard[0, 0, :, :, :] * 0.5)

    H_seg, V_seg, _, _, _ = temp_hexaboard.shape
    threshold = np.full((H_seg, V_seg), 0.1, dtype=float)

    flagged_segments = pixelwise_inference(temp_hexaboard, new_hexaboard, threshold, skipped_segments=set())
    assert len(flagged_segments) >= 1, "At least one segment should be flagged for different images."
    assert (0, 0) in flagged_segments, "The modified segment should be flagged."


def test_identical_autoencoder_inference(baseline_hexaboard: np.ndarray, config: Tuple) -> None:
    _, height, width, latent_dim, init_filters, layers = config
    ckpt_path = LOG_DIR + '/CNNAutoencoder/best/run_01.pt'

    if os.path.exists(ckpt_path):
        H_seg, V_seg, height, width, _ = baseline_hexaboard.shape
        model = CNNAutoencoder(
            height=height,
            width=width,
            latent_dim=latent_dim,
            init_filters=init_filters,
            layers=layers
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        threshold = np.full((H_seg, V_seg), 1.0, dtype=float)  # 2D threshold array

        flagged_segments = autoencoder_inference(
            hexaboard=baseline_hexaboard,
            threshold=threshold,
            model=model,
            device=device
        )

        assert len(flagged_segments) == 0, "No segments should be flagged for very low threshold."
    else:
        pytest.skip(f"Model checkpoint not found at {ckpt_path}")


def test_pixelwise_inference_shape_mismatch() -> None:
    baseline = np.random.rand(13, 9, 100, 100, 3).astype(np.uint8)
    new = np.random.rand(10, 9, 100, 100, 3).astype(np.uint8)  # different H_seg
    threshold = np.zeros((13, 9), dtype=float)  # 2D threshold array
    
    with pytest.raises(ValueError, match="Baseline and new hexaboards must have the same shape."):
        pixelwise_inference(baseline, new, threshold)


def test_autoencoder_inference_wrong_dims(baseline_hexaboard: np.ndarray, config: Tuple) -> None:
    _, height, width, latent_dim, init_filters, layers = config
    ckpt_path = LOG_DIR + '/CNNAutoencoder/best/run_01.pt'

    if os.path.exists(ckpt_path):
        H_seg, V_seg, height, width, _ = baseline_hexaboard.shape
        model = CNNAutoencoder(
            height=height,
            width=width,
            latent_dim=latent_dim,
            init_filters=init_filters,
            layers=layers
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        threshold = np.zeros((H_seg, V_seg), dtype=float)  # 2D threshold array
        wrong_shape = np.random.rand(100, 100, 3).astype(np.uint8)  # 3D instead of 5D
    
        with pytest.raises(ValueError, match="Expected 5D array for hexaboard, got 3D array instead."):
            autoencoder_inference(wrong_shape, threshold, model, device)
    else:
        pytest.skip(f"Model checkpoint not found at {ckpt_path}")