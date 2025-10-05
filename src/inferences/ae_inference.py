from typing import List, Set, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def autoencoder_inference(
    hexaboard: np.ndarray,
    threshold: np.ndarray,
    model: nn.Module,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    skipped_segments: Set[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """
    Compares the autoencoder's reconstruction of hexaboard segments against the original segments and
    flags every (H_seg_idx, V_seg_idx) segment whose reconstruction SSIM drops below *threshold*.

    Parameters
    ----------
    hexaboard: np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing a single hexaboard.
    threshold: float
        The SSIM threshold below which segments are flagged.
    model: nn.Module
        The autoencoder model to use for inference.
    device: torch.device
        The device to run the inference on.
    skipped_segments: Set[Tuple[int, int]], optional
        A set of (H_seg, V_seg) tuples representing the segments to skip.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing the (H_seg, V_seg) indices of the flagged segments.
    """
    if hexaboard.ndim != 5:
        raise ValueError(f"Expected 5D array for hexaboard, got {hexaboard.ndim}D array instead.")

    H_seg, V_seg, _, _, _ = hexaboard.shape
    
    # Set the model to evaluation mode
    model.eval()

    # Create dataset
    hexaboard = torch.tensor(hexaboard).permute(0, 1, 4, 2, 3)
    flagged_segments = []
    if skipped_segments is None:
        skipped_segments = {
            (0, 0), (0, 1), (0, 7), (0, 8),
            (1, 0), (1, 8),
            (2, 0), (2, 8),
            (3, 0), (3, 8),
            (4, 0), (4, 8),
            (8, 0), (8, 8),
            (9, 0), (9, 8),
            (10, 0), (10, 1), (10, 7), (10, 8),
            (11, 0), (11, 1), (11, 8),
            (12, 0), (12, 1), (12, 7), (12, 8)
        }

    for h in range(H_seg):
        for v in range(V_seg):
            if (h, v) in skipped_segments:
                continue

            segment = hexaboard[h, v].unsqueeze(0)  # (1, num_channels, height, width)
            segment = segment.to(device, non_blocking=device.type == 'cuda')
            recons = torch.sigmoid(model(segment))
            pred = recons[0].permute(1, 2, 0)
            true = segment[0].permute(1, 2, 0)
            mae_val = F.l1_loss(pred, true).item()
            if mae_val >= threshold[h, v]:
                flagged_segments.append((h, v))

    return flagged_segments