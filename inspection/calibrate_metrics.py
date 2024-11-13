# Import necessary dependencies
from typing import List, Tuple
from PIL import Image

import numpy as np
from inspection.calibrate_utils import evaluate_inspection, process_inspection

# General function to evaluate SSIM for good and bad segments
# needs to be renamed to pw_calibrate_metrics
def calibrate_metrics(
    segments1: List[Image.Image],
    segments2: List[Image.Image],
    differences: List[int]
) -> Tuple[float, List, List]:
    """
    Calibrates inspection metrics by evaluating the SSIM for good and bad segments and determining an optimal threshold.

    Args:

        segments1 (List[Image.Image]): List of image segments from the first set.
        segments2 (List[Image.Image]): List of image segments from the second set.
        differences (List[int]): Indices indicating segments that have differences.

    Returns:
        (optimal_threshold, bad_values, good_values) (Tuple[float, List, List]):
            - Optimal SSIM threshold for classifying segments.
            - List of bad SSIM values and their corresponding segment pairs.
            - List of good SSIM values and their corresponding segment pairs.
    """
    bad_ssims = []
    good_ssims = []
    bad_segments = []
    good_segments = []

    for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):
        segment1, segment2 = np.array(segment1), np.array(segment2)
        measure_value = evaluate_inspection(segment1, segment2)

        if i in differences:
            bad_ssims.append(measure_value)
            bad_segments.append((segment1, segment2))
        else:
            good_ssims.append(measure_value)
            good_segments.append((segment1, segment2))

    # Process the inspection to find the optimal SSIM threshold
    optimal_threshold = process_inspection(np.array(good_ssims), np.array(bad_ssims))
    
    # Pair the SSIM values with their respective segments
    bad_values = list(zip(bad_ssims, bad_segments))
    good_values = list(zip(good_ssims, good_segments))

    return optimal_threshold, bad_values, good_values