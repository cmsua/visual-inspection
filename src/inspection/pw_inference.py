from typing import List, Set, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..utils.data import load_skipped_segments
from ..utils.data.board_contract import validate_coordinates


def pixelwise_inference(
    baseline_hexaboard: np.ndarray,
    new_hexaboard: np.ndarray,
    threshold: np.ndarray,
    skipped_segments: Set[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """
    Performs pixel-wise comparison between the baseline and new hexaboard images and
    flags every (H_seg_idx, V_seg_idx) segment whose (1 - SSIM) is above `threshold`.

    Parameters
    ----------
    baseline_hexaboard : np.ndarray
        5D array with shape `[H_seg, V_seg, height, width, num_channels]` representing the baseline hexaboard.
    new_hexaboard : np.ndarray
        5D array with shape `[H_seg, V_seg, height, width, num_channels]` representing the new hexaboard to compare.
    threshold : np.ndarray
        The SSIM threshold below which segments are flagged.
    skipped_segments : Set[Tuple[int, int]], optional
        A set of (H_seg, V_seg) tuples representing the segments to skip.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing the (H_seg, V_seg) indices of the flagged segments.
    """
    if baseline_hexaboard.shape != new_hexaboard.shape:
        raise ValueError("Baseline and new hexaboards must have the same shape.")

    H_seg, V_seg, _, _, _ = baseline_hexaboard.shape
    flagged_segments = []
    if skipped_segments is None:
        skipped_segments = load_skipped_segments(shape=(H_seg, V_seg))

    skipped_segments = validate_coordinates(skipped_segments, (H_seg, V_seg))
    if threshold.shape != (H_seg, V_seg) or not np.isfinite(threshold).all():
        raise ValueError("Threshold shape must match the grid and contain finite scores.")

    # Loop over all horizontal and vertical segments
    for h in range(H_seg):
        for v in range(V_seg):
            if (h, v) in skipped_segments:
                continue

            baseline_segment = baseline_hexaboard[h, v]
            new_segment = new_hexaboard[h, v]
            ssim_val = 1 - ssim(baseline_segment, new_segment, data_range=1.0, channel_axis=2)
            if ssim_val >= threshold[h, v]:
                flagged_segments.append((h, v))

    return flagged_segments
