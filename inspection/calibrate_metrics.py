# Import necessary dependencies
from typing import List, Tuple
import json
import argparse
from PIL import Image

import numpy as np
from skimage.metrics import structural_similarity as ssim

from inspection.calibrate_utils import process_inspection

# General function to obtain calibration metrics (optimal threshold for SSIM)
def calibrate_metrics(
    baseline_segments: List[Image.Image],
    perturbed_segments: List[Image.Image],
    second_baseline_segments: List[Image.Image]
) -> Tuple[float, List[float], List[float]]:
    """
    Calibrates metrics by calculating SSIM values for pairs of image segments, distinguishing
    between 'bad' segments (perturbed vs. baseline) and 'good' segments (baseline vs. second baseline).
    Determines the optimal SSIM threshold for segment classification.

    Args:
        baseline_segments (List[Image.Image]): List of image segments representing the baseline state.
        perturbed_segments (List[Image.Image]): List of image segments representing a perturbed state.
        second_baseline_segments (List[Image.Image]): List of image segments representing a second baseline for comparison.

    Returns:
        (optimal_threshold, bad_ssims, good_ssims) (Tuple[float, List[float], List[float]]):
            - Optimal SSIM threshold for distinguishing good and bad segments.
            - List of SSIM values calculated for baseline vs. perturbed segments (bad SSIMs).
            - List of SSIM values calculated for baseline vs. second baseline segments (good SSIMs).
    """
    bad_ssims = []
    good_ssims = []

    for i, (seg1, seg2) in enumerate(zip(baseline_segments, perturbed_segments)):
        seg1, seg2 = np.array(seg1), np.array(seg2)
        measure_value = ssim(seg1, seg2, channel_axis=2)

        bad_ssims.append(measure_value)

    for i, (seg1, seg2) in enumerate(zip(baseline_segments, second_baseline_segments)):
        seg1, seg2 = np.array(seg1), np.array(seg2)
        measure_value = ssim(seg1, seg2, channel_axis=2)

        good_ssims.append(measure_value)

    # Process the inspection to find the optimal SSIM threshold
    optimal_threshold = process_inspection(np.array(good_ssims), np.array(bad_ssims))

    return optimal_threshold, bad_ssims, good_ssims

# SSIM metric calibration
# RUN THIS COMMAND FROM visual-inspection
# python -m inspection.calibrate_metrics -n "datasets/raw_images/hexaboard_01.png" -b "datasets/raw_images/hexaboard_02.png" -s "datasets/raw_images/hexaboard_02.png" -vs 20 -hs 12
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Visual Inspection',
        description='Inspects Hexaboards for defects',
        epilog='University of Alabama'
    )

    # Set up arguments
    parser.add_argument('-n', '--new_images_path', type=str, help='path to new image')
    parser.add_argument('-b', '--baseline_images_path', type=str, help='path to baseline image')
    parser.add_argument('-s', '--second_baseline_images_path', type=str, help='path to second baseline image')
    parser.add_argument('-vs', '--vertical_segments', type=int, help='number of vertical image segments')
    parser.add_argument('-hs', '--horizontal_segments', type=int, help='number of horizontal image segments')
    
    # Parse
    args = parser.parse_args()
    new_images = np.load(args.new_images_path)
    baseline_images = np.load(args.baseline_images_path)
    second_baseline_images = np.load(args.baseline_image_path)

    # Calibrate the threshold
    threshold, bad_ssims, good_ssims = calibrate_metrics(baseline_images, new_images, second_baseline_images)

    # Save to a JSON file
    with open('inspection/calibration.json', 'w') as fout:
        json.dump([threshold, bad_ssims, good_ssims], fout)