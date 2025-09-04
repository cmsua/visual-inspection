import json
from typing import Tuple, Optional
from collections import defaultdict

import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_curve


def process_inspection(good_ssims: np.ndarray, bad_ssims: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Processes SSIM arrays to compute an optimal threshold using ROC.

    Parameters
    ----------
    good_ssims : np.ndarray
        2D array of SSIM values for good image segments.
    bad_ssims : np.ndarray
        2D array of SSIM values for bad image segments.
    mask : np.ndarray, optional
        Optional boolean mask (same shape as inputs) selecting valid positions.

    Returns
    -------
    float
        The optimal SSIM threshold that minimizes the false negative rate (1 - TPR).
    """
    good = np.asarray(good_ssims)
    bad = np.asarray(bad_ssims)

    if good.shape != bad.shape:
        raise ValueError(f"good_ssims and bad_ssims must have the same shape, got {good.shape} vs. {bad.shape}.")

    if mask is not None:
        if mask.shape != good.shape:
            raise ValueError(f"mask shape {mask.shape} must match SSIM shape {good.shape}.")
        good_vals = good[mask]
        bad_vals = bad[mask]
    else:
        # Default: drop uncomputed elements
        good_vals = good.ravel()
        bad_vals = bad.ravel()
        mask = bad_vals > 0
        good_vals = good_vals[mask]
        bad_vals = bad_vals[mask]

    if good_vals.size == 0 or bad_vals.size == 0:
        raise ValueError("No valid SSIM values found for good or bad segments.")

    ssim_values = np.concatenate([good_vals, bad_vals])
    labels = np.concatenate([
        np.ones_like(good_vals, dtype=np.int8),
        np.zeros_like(bad_vals, dtype=np.int8),
    ])

    fpr, tpr, roc_thresholds = roc_curve(labels, ssim_values)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(roc_thresholds[optimal_idx])

    return optimal_threshold


def calibrate_metrics(
    baseline_hexaboard_path: str = './data/train/aligned_images1.npy',
    good_hexaboard_path: str = './data/train/aligned_images2.npy',
    json_map: str = 'damaged_segments.json',
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calibrates metrics by calculating SSIM values for pairs of hexaboard segments, distinguishing
    between 'bad' segments (bad vs. baseline) and 'good' segments (baseline vs. good).
    Determines the optimal SSIM threshold for segment classification.

    Parameters
    ----------
    baseline_hexaboard_path : str
        Path to the file containing the baseline hexaboard.
    good_hexaboard_path : str
        Path to the file containing the good hexaboard.
    json_map: str
        Path to the JSON file containing indices of the bad segments.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - optimal_threshold (np.ndarray): The optimal SSIM threshold for segment classification.
        - bad_ssims (np.ndarray): SSIM values for 'bad' segments (perturbed vs. baseline).
        - good_ssims (np.ndarray): SSIM values for 'good' segments (baseline vs. second baseline).
    """
    # Load the baseline and good hexaboards
    baseline_hexaboard = np.load(baseline_hexaboard_path)
    baseline_hexaboard = baseline_hexaboard[..., ::-1].copy()
    good_hexaboard = np.load(good_hexaboard_path)
    good_hexaboard = good_hexaboard[..., ::-1].copy()

    if baseline_hexaboard.ndim != 5 or good_hexaboard.ndim != 5:
        raise ValueError("All hexaboard arrays must be 5D (H_seg, V_seg, height, width, num_channels).")
    
    H_seg, V_seg, _, _, _ = baseline_hexaboard.shape

    # Normalize to [0, 1] range
    baseline_hexaboard = baseline_hexaboard / 255.0
    good_hexaboard = good_hexaboard / 255.0

    # Open the JSON file to get a list of all indices of bad segments
    if json_map is not None:
        with open(json_map, 'r') as f:
            bad_segments_mapping = json.load(f)['files']

    # Create a dictionary to map bad segment indices to their corresponding hexaboard segments
    bad_segments_dict = defaultdict(list)

    for filename in bad_segments_mapping.keys():
        image = np.load(filename)
        image = image[..., ::-1].copy()

        for coord in bad_segments_mapping[filename]['damaged']:
            h, v = coord['row'], coord['col']
            bad_segments_dict[(h, v)].append(image[h, v])

    bad_ssims = np.zeros((H_seg, V_seg))
    good_ssims = np.zeros((H_seg, V_seg))
    mask = np.zeros((H_seg, V_seg), dtype=bool)

    # Calculate SSIM
    for (h, v) in bad_segments_dict.keys():
        baseline_segment = baseline_hexaboard[h, v]
        good_segment = good_hexaboard[h, v]
        mask[h, v] = True
        good_ssims[h, v] = ssim(baseline_segment, good_segment, data_range=1.0, channel_axis=2)
        same_segment_ssim = []

        for bad_segment in bad_segments_dict[(h, v)]:
            same_segment_ssim.append(ssim(baseline_segment, bad_segment, data_range=1.0, channel_axis=2))

        bad_ssims[h, v] = np.mean(same_segment_ssim) if same_segment_ssim else 0

    # Process the inspection to find the optimal SSIM threshold
    optimal_threshold = process_inspection(good_ssims, bad_ssims, mask)

    return optimal_threshold, bad_ssims, good_ssims