import os
import tempfile

import pytest
import numpy as np

import torch

from src.inferences import autoencoder_inference, pixelwise_inference

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = './data/train'
LOG_DIR = './logs'


@pytest.fixture
def baseline_hexaboard() -> np.ndarray:
    """Load a baseline hexaboard as 5D array."""
    # Load the reference data - should be 5D
    data = np.load(DATA_DIR + '/ref_image_array.npy')
    if data.ndim == 6:
        # If it's 6D, take the first board
        data = data[0]
    return data


@pytest.fixture
def temp_hexaboard() -> np.ndarray:
    """Create a temporary hexaboard for testing."""
    # Create a temporary 5D array with random data
    temp_data = np.random.rand(12, 9, 100, 100, 3) * 255  # Smaller size for faster tests
    return temp_data.astype(np.uint8)


def test_identical_pixelwise_inference(temp_hexaboard: np.ndarray) -> None:
    flagged_segments = pixelwise_inference(temp_hexaboard, temp_hexaboard, threshold=0.9)
    assert len(flagged_segments) == 0, "No segments should be flagged for identical images."


def test_one_diff_pixelwise_inference(temp_hexaboard: np.ndarray) -> None:
    # Create a modified version
    new_hexaboard = temp_hexaboard.copy()
    new_hexaboard[0, 0, :, :, :] = (new_hexaboard[0, 0, :, :, :] * 0.5).astype(np.uint8)  # significant change
    
    flagged_segments = pixelwise_inference(temp_hexaboard, new_hexaboard, threshold=0.9)
    assert len(flagged_segments) >= 1, "At least one segment should be flagged for different images."
    assert (0, 0) in flagged_segments, "The modified segment should be flagged."


def test_identical_autoencoder_inference(baseline_hexaboard: np.ndarray) -> None:
    ckpt_path = LOG_DIR + '/ResNetAutoencoder/best/run_01.pt'
    if os.path.exists(ckpt_path):
        flagged_segments = autoencoder_inference(
            hexaboard=baseline_hexaboard,
            threshold=0.0,
            device=device,
            best_model_path=ckpt_path
        )
        assert len(flagged_segments) == 0, "No segments should be flagged for very low threshold."
    else:
        pytest.skip(f"Model checkpoint not found at {ckpt_path}")


def test_pixelwise_inference_shape_mismatch():
    """Test that pixelwise inference raises error for mismatched shapes."""
    baseline = np.random.rand(12, 9, 100, 100, 3).astype(np.uint8)
    new = np.random.rand(10, 9, 100, 100, 3).astype(np.uint8)  # different H_seg
    
    with pytest.raises(ValueError, match="must have the same shape"):
        pixelwise_inference(baseline, new, threshold=0.5)


def test_autoencoder_inference_wrong_dims():
    """Test that autoencoder inference raises error for wrong dimensions."""
    wrong_shape = np.random.rand(100, 100, 3).astype(np.uint8)  # 3D instead of 5D
    
    with pytest.raises(ValueError, match="Expected 5D array"):
        autoencoder_inference(wrong_shape, threshold=0.5)