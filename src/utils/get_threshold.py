import json
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn.functional as F
from torch import nn

from .data import load_hexaboard


def process_metrics(
    good_ssims: np.ndarray,
    bad_ssims: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fill_mode: str = 'min'
) -> np.ndarray:
    """
    Compute a per-segment SSIM and MAE threshold map.

    For each segment (h, v) present in `mask` (or inferred from `bad_ssims` if mask is None)
    compute a threshold. Since `good_ssims` and `bad_ssims` contain a single SSIM
    value per segment (or an average), we use the midpoint between the good and bad
    SSIM values as the per-segment threshold. Cells not present in `mask` are filled
    with the mean of the computed thresholds.

    Parameters
    ----------
    good_ssims: np.ndarray
        2D array of SSIM values for good image segments.
    bad_ssims: np.ndarray
        2D array of SSIM values for bad image segments.
    mask: np.ndarray, optional
        Optional boolean mask (same shape as inputs) selecting valid positions.
    fill_mode: str
        Method to fill segments not in `mask` or where computation failed:
        `min` (default) uses the minimum computed threshold,
        `max` uses the maximum computed threshold,
        any other value uses the mean computed threshold.

    Returns
    -------
    np.ndarray
        Array of shape (H_seg, V_seg) with a threshold value for each segment.
    """
    good = np.asarray(good_ssims, dtype=float)
    bad = np.asarray(bad_ssims, dtype=float)

    if good.shape != bad.shape:
        raise ValueError(f"good_ssims and bad_ssims must have the same shape, got {good.shape} vs. {bad.shape}.")

    H, W = good.shape

    if mask is None:
        # Infer mask from bad_ssims: any finite, non-zero value indicates the segment was listed
        mask = np.isfinite(bad) & (bad != 0)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (H, W):
        raise ValueError(f"mask shape {mask.shape} must match SSIM shape {(H, W)}.")

    thresholds = np.full((H, W), np.nan, dtype=float)

    # For segments in the mask compute threshold as midpoint (good + bad) / 2
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue

            g = good[i, j]
            b = bad[i, j]
            if not (np.isfinite(g) and np.isfinite(b)):
                thresholds[i, j] = np.nan
            else:
                thresholds[i, j] = b  # use min of MAEs and SSIMs for guaranteed true positive

    # Compute fill value as mean of computed thresholds for masked segments
    computed = np.isfinite(thresholds) & mask
    if not computed.any():
        raise ValueError("No valid thresholds could be computed for any masked segment.")

    if fill_mode == 'min':
        fill_value = float(np.nanmin(thresholds[computed]))
    elif fill_mode == 'max':
        fill_value = float(np.nanmax(thresholds[computed]))
    else:
        fill_value = float(np.nanmean(thresholds[computed]))

    # Fill segments not in mask (or where computation failed) with the mean value
    thresholds[~computed] = fill_value

    return thresholds


def calibrate_metrics(
    baseline_hexaboard: np.ndarray,
    good_hexaboard_paths: List[str],
    model: nn.Module,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    json_map_path: str = './calibrations/damaged_segments.json',
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Calibrates metrics by calculating SSIM and MAE values for pairs of hexaboard segments, distinguishing
    between 'bad' segments (bad vs. baseline) and 'good' segments (baseline vs. good).
    Determines the optimal SSIM threshold for pixel-wise comparison and autoencoder reconstruction.

    Parameters
    ----------
    baseline_hexaboard: np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing the baseline hexaboard.
    good_hexaboard: np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing the second good hexaboard to compare.
    model: nn.Module
        The autoencoder model used for reconstructing the segments.
    device: torch.device
        The device to run the calibration on.
    json_map_path: str
        Path to the JSON file containing indices of the bad segments.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - optimal_threshold (np.ndarray): The optimal SSIM threshold for segment classification.
        - bad_ssims (np.ndarray): SSIM values for 'bad' segments (baseline vs. perturbed).
        - good_ssims (np.ndarray): SSIM values for 'good' segments (baseline vs. second baseline).
    """
    H_seg, V_seg, _, _, _ = baseline_hexaboard.shape

    # Set the model to evaluation mode
    model.eval()

    # Open the JSON file to get a list of all indices of bad segments
    if json_map_path is not None:
        with open(json_map_path, 'r') as f:
            bad_segments_mapping = json.load(f)['files']

    # Create a dictionary to keep track of which good segments are in the same position as bad segments
    good_segments_dict = defaultdict(list)
    for filename in good_hexaboard_paths:
        image = load_hexaboard(filename)
        for info_dict in bad_segments_mapping.values():
            for coord in info_dict['damaged']:
                h, v = coord['row'], coord['col']
                good_segments_dict[(h, v)].append(image[h, v])

    # Create a dictionary to map bad segment indices to their corresponding hexaboard segments
    bad_segments_dict = defaultdict(list)
    for filename in bad_segments_mapping.keys():
        image = load_hexaboard(filename)
        for coord in bad_segments_mapping[filename]['damaged']:
            h, v = coord['row'], coord['col']
            bad_segments_dict[(h, v)].append(image[h, v])

    # Initialize arrays to store SSIM values for autoencoder and pixel-wise comparisons
    pw_good_ssims = np.zeros((H_seg, V_seg))
    ae_good_maes = np.zeros((H_seg, V_seg))
    pw_bad_ssims = np.zeros((H_seg, V_seg))
    ae_bad_maes = np.zeros((H_seg, V_seg))
    mask = np.zeros((H_seg, V_seg), dtype=bool)

    # Calculate SSIM and MAE for each good segment against the baseline
    pw_same_segment_good_ssim = []
    ae_same_segment_good_mae = []
    for (h, v), good_segments in good_segments_dict.items():
        baseline_segment = baseline_hexaboard[h, v]

        pw_same_segment_good_ssim = []
        ae_same_segment_good_mae = []
        for good_segment in good_segments:
            # Pixel-wise SSIM on good segments
            pw_same_segment_good_ssim.append(1 - ssim(baseline_segment, good_segment, data_range=1.0, channel_axis=2))

            # Autoencoder MAE on good segments
            segment = torch.tensor(good_segment).permute(2, 0, 1).unsqueeze(0)  # (1, num_channels, height, width)
            segment = segment.to(device, non_blocking=device.type == 'cuda')
            recons = torch.sigmoid(model(segment))
            pred = recons[0].permute(1, 2, 0)
            true = segment[0].permute(1, 2, 0)
            ae_same_segment_good_mae.append(F.l1_loss(pred, true).item())

        pw_good_ssims[h, v] = np.max(pw_same_segment_good_ssim) if pw_same_segment_good_ssim else 0
        ae_good_maes[h, v] = np.max(ae_same_segment_good_mae) if ae_same_segment_good_mae else 0
        
    # Calculate SSIM and MAE for each bad segment against the baseline
    for (h, v), bad_segments in bad_segments_dict.items():
        baseline_segment = baseline_hexaboard[h, v]
        mask[h, v] = True

        pw_same_segment_bad_ssim = []
        ae_same_segment_bad_mae = []
        for bad_segment in bad_segments:
            # Pixel-wise SSIM on bad segments
            pw_same_segment_bad_ssim.append(1 - ssim(baseline_segment, bad_segment, data_range=1.0, channel_axis=2))

            # Autoencoder MAE on bad segments
            segment = torch.tensor(bad_segment).permute(2, 0, 1).unsqueeze(0)  # (1, num_channels, height, width)
            segment = segment.to(device, non_blocking=device.type == 'cuda')
            recons = torch.sigmoid(model(segment))
            pred = recons[0].permute(1, 2, 0)
            true = segment[0].permute(1, 2, 0)
            ae_same_segment_bad_mae.append(F.l1_loss(pred, true).item())

        if len(pw_same_segment_bad_ssim) > 1:
            print(f"Multiple bad segments at ({h}, {v}).")

        pw_bad_ssims[h, v] = np.min(pw_same_segment_bad_ssim) if pw_same_segment_bad_ssim else 0
        ae_bad_maes[h, v] = np.min(ae_same_segment_bad_mae) if ae_same_segment_bad_mae else 0

    # Process the inspection to find the optimal SSIM threshold for each method
    pw_threshold = process_metrics(pw_good_ssims, pw_bad_ssims, mask, fill_mode='min')
    ae_threshold = process_metrics(ae_good_maes, ae_bad_maes, mask, fill_mode='min')

    pw_metrics = (pw_threshold, pw_bad_ssims, pw_good_ssims)
    ae_metrics = (ae_threshold, ae_bad_maes, ae_good_maes)

    return pw_metrics, ae_metrics