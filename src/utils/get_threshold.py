import json
from typing import Tuple, Optional
from collections import defaultdict

import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_curve


def process_inspection(good_ssims: np.ndarray, bad_ssims: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute a per-segment SSIM threshold map.

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
        # infer mask from bad_ssims: any finite, non-zero value indicates the segment was listed
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
                # If either value is missing/invalid, leave as NaN for now
                thresholds[i, j] = np.nan
            else:
                thresholds[i, j] = float((g + b) / 2.0)

    # Compute fill value as mean of computed thresholds for masked segments
    computed = np.isfinite(thresholds) & mask
    if not computed.any():
        raise ValueError("No valid thresholds could be computed for any masked segment.")

    fill_value = float(np.nanmean(thresholds[computed]))

    # Fill segments not in mask (or where computation failed) with the mean value
    thresholds[~computed] = fill_value

    return thresholds


def calibrate_metrics(
    baseline_hexaboard_path: str = './data/train/aligned_images1.npy',
    good_hexaboard_path: str = './data/train/aligned_images2.npy',
    json_map: str = './calibrations/damaged_segments.json',
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calibrates metrics by calculating SSIM values for pairs of hexaboard segments, distinguishing
    between 'bad' segments (bad vs. baseline) and 'good' segments (baseline vs. good).
    Determines the optimal SSIM threshold for segment classification.

    Parameters
    ----------
    baseline_hexaboard_path: str
        Path to the file containing the baseline hexaboard.
    good_hexaboard_path: str
        Path to the file containing the good hexaboard.
    json_map: str
        Path to the JSON file containing indices of the bad segments.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - optimal_threshold (np.ndarray): The optimal SSIM threshold for segment classification.
        - bad_ssims (np.ndarray): SSIM values for 'bad' segments (baseline vs. perturbed).
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
        image = image / 255.0

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

        if len(same_segment_ssim) > 1:
            print(f"Multiple bad segments at ({h}, {v}).")

        bad_ssims[h, v] = np.mean(same_segment_ssim) if same_segment_ssim else 0

    # Process the inspection to find the optimal SSIM threshold
    optimal_threshold = process_inspection(good_ssims, bad_ssims, mask)

    return optimal_threshold, bad_ssims, good_ssims