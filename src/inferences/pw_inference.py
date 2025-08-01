from typing import List, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim


def pixelwise_inference(
    baseline_hexaboard: np.ndarray,
    new_hexaboard: np.ndarray,
    threshold: float
) -> List[Tuple[int, int]]:
    """
    Performs pixel-wise comparison between the baseline and new hexaboard images and
    flags every (H_seg_idx, V_seg_idx) segment whose SSIM drops below *threshold*.

    Parameters
    ----------
    baseline_hexaboard : np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing the baseline hexaboard.
    new_hexaboard : np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing the new hexaboard to compare.
    threshold : float
        The SSIM threshold below which segments are flagged.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing the (H_seg_idx, V_seg_idx) indices of the flagged segments.
    """

    if baseline_hexaboard.shape != new_hexaboard.shape:
        raise ValueError(f"Baseline and new hexaboards must have the same shape.")

    H_seg, V_seg, _, _, _ = baseline_hexaboard.shape
    baseline_hexaboard = baseline_hexaboard.astype(np.float32) / 255.0
    new_hexaboard = new_hexaboard.astype(np.float32) / 255.0

    flagged_segments = []

    # Loop over all horizontal and vertical segments
    for h in range(H_seg):
        for v in range(V_seg):
            baseline_segment = baseline_hexaboard[h, v]
            new_segment = new_hexaboard[h, v]
            ssim_val = ssim(baseline_segment, new_segment, data_range=1.0, channel_axis=2)

            if ssim_val < threshold:
                flagged_segments.append((h, v))

    return flagged_segments