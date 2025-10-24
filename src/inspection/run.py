from typing import List

import numpy as np

import torch

from .ae_inference import autoencoder_inference
from .pw_inference import pixelwise_inference
from ..models import CNNAutoencoder
from ..utils import InspectionResults, set_seed
from ..utils.data import load_hexaboard, load_skipped_segments


def run_inspection(
    baseline_hexaboard_path: str,
    new_hexaboard_path: str,
    skipped_segments_path: str,
    ae_threshold_path: str = './calibrations/ae_threshold.npy',
    pw_threshold_path: str = './calibrations/pw_threshold.npy',
    latent_dim: int = 32,
    init_filters: int = 128,
    layers: List = [2, 2, 2],
    best_model_path: str = './logs/CNNAutoencoder/best/run_01.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> InspectionResults:
    """
    Run visual inspection between two hexaboards using autoencoder and pixel-wise methods.

    Parameters
    ----------
    baseline_hexaboard_path: str
        Path to the baseline hexaboard.
    new_hexaboard_path: str
        Path to the new hexaboard to inspect.
    skipped_segments_path: str
        Path to the JSON file containing the list of segments to skip.
    ae_threshold_path: str
        Path to the autoencoder threshold .npy file.
    pw_threshold_path: str
        Path to the pixel-wise threshold .npy file.
    latent_dim: int
        Bottleneck dimension for the autoencoder.
    init_filters: int
        Initial number of filters in the model.
    layers: List[int]
        Number of CNN stages and their blocks.
    best_model_path: str
        Path to the best model weights.
    device: str
        Device to use for inference.

    Returns
    -------
    InspectionResults
        The results of the visual inspection.
    """
    device = torch.device(device)

    # Reproducibility settings
    set_seed(42)

    # Load the hexaboard data
    baseline_hexaboard = load_hexaboard(baseline_hexaboard_path)
    new_hexaboard = load_hexaboard(new_hexaboard_path)
    H_seg, V_seg, height, width, channels = baseline_hexaboard.shape
    segment_shape = (height, width, channels)
    grid_shape = (H_seg, V_seg)

    # Load the model
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load the thresholds
    ae_threshold = np.load(ae_threshold_path)
    pw_threshold = np.load(pw_threshold_path)

    # Load the skipped segments
    skipped_segments = load_skipped_segments(skipped_segments_path)

    # Perform inferences
    ae_indices = autoencoder_inference(
        hexaboard=new_hexaboard,
        threshold=ae_threshold,
        model=model,
        device=device,
        skipped_segments=skipped_segments
    )
    pw_indices = pixelwise_inference(
        baseline_hexaboard=baseline_hexaboard,
        new_hexaboard=new_hexaboard,
        threshold=pw_threshold,
        skipped_segments=skipped_segments
    )

    # Compile the inspection results
    results = InspectionResults.from_segment_flags(
        shape=grid_shape,
        pixel_flagged=pw_indices,
        autoencoder_flagged=ae_indices,
        skipped_segments=skipped_segments,
        baseline_path=baseline_hexaboard_path,
        inspected_path=new_hexaboard_path,
        segment_shape=segment_shape
    )

    return results