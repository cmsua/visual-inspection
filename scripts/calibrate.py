import argparse

import numpy as np

from src.utils import calibrate_metrics
from src.utils.viz import plot_threshold_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate SSIM threshold for each segment.")

    # Data I/O arguments
    parser.add_argument('--baseline-hexaboard', type=str, default='./data/train/aligned_images1.npy', help="Path to the baseline hexaboard")
    parser.add_argument('--good-hexaboard', type=str, default='./data/train/aligned_images2.npy', help="Path to the good hexaboard")
    parser.add_argument('--json-map', type=str, default='./calibrations/damaged_segments.json', help="Path to the JSON map file")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Calibrate metrics and get SSIM thresholds
    optimal_threshold, bad_ssims, good_ssims = calibrate_metrics(
        baseline_hexaboard_path=args.baseline_hexaboard,
        good_hexaboard_path=args.good_hexaboard,
        json_map=args.json_map,
    )

    # Visualize the threshold comparison
    plot_threshold_comparison(
        optimal_threshold=optimal_threshold,
        bad_ssims=bad_ssims,
        good_ssims=good_ssims
    )

    # Save the results to .npy files
    np.save('./calibrations/optimal_threshold.npy', optimal_threshold)
    np.save('./calibrations/bad_ssims.npy', bad_ssims)
    np.save('./calibrations/good_ssims.npy', good_ssims)


if __name__ == '__main__':
    main()