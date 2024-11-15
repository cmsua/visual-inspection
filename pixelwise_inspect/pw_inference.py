# Import necessary dependencies
from typing import List
from PIL import Image

import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function for pixel-by-pixel comparison
def pw_inference(
    new_segments: List[Image.Image],
    baseline_segments: List[Image.Image],
    threshold: float
) -> List[int]:
    """
    Performs pixel-wise inference by comparing SSIM (Structural Similarity Index) values between corresponding
    segments from two images and identifying those below a specified threshold.

    Args:
        new_segments (List[Image.Image]): List of segmented images (new segments) to be inspected.
        baseline_segments (List[Image.Image]): List of segmented baseline images for comparison.
        threshold (float): SSIM threshold below which segments are flagged as differing significantly.

    Returns:
        flagged_indices (List[int]): List of indices where the SSIM between corresponding segments is below the threshold.
    """
    assert len(new_segments) == len(baseline_segments), "Segment lists are not of the same length"

    flagged_indices = []

    # Iterate over all segments
    for i, (seg1, seg2) in enumerate(zip(new_segments, baseline_segments)):
        img1 = np.array(seg1)
        img2 = np.array(seg2)

        # Check if the segment shapes are different
        assert img1.shape == img2.shape, 'Image 1 and Image 2 in compare_segments have different shapes'
    
        # SSIM metrics
        ssim_val = ssim(img1, img2, channel_axis=2)

        # Filter the indices
        if ssim_val < threshold:
            flagged_indices.append(i)

    return flagged_indices