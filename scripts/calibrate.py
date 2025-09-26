import argparse
from typing import List

import numpy as np

import torch

from src.models import CNNAutoencoder
from src.utils import calibrate_metrics, set_seed
from src.utils.data import load_hexaboard, load_skipped_segments
from src.utils.viz import plot_threshold_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate SSIM threshold for each segment.")

    # Data I/O arguments
    parser.add_argument('-b', '--baseline-hexaboard-path', type=str, default='./data/train/aligned_images1.npy', help="Path to the baseline hexaboard")
    parser.add_argument('-g', '--good-hexaboard-path', type=str, default='./data/train/aligned_images2.npy', help="Path to the good hexaboard")
    parser.add_argument('-j', '--json-map-path', type=str, default='./calibrations/damaged_segments.json', help="Path to the JSON map file")
    parser.add_argument('-s', '--skipped-segments-path', type=str, default='./calibrations/skipped_segments.json', help="Path to the JSON file containing the list of segments to skip")

    # Model architecture arguments
    parser.add_argument('--latent-dim', type=int, default=32, help="Bottleneck dimension")
    parser.add_argument('--init-filters', type=int, default=128, help="Initial number of filters in the model")
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 2, 2], help="Number of CNN stages and their blocks")
    parser.add_argument('-w', '--best-model-path', type=str, default='./logs/CNNAutoencoder/best/run_01.pt', help="Path to the best model weights")

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for calibration")

    return parser.parse_args()


def main(
    baseline_hexaboard_path: str,
    good_hexaboard_path: str,
    json_map_path: str,
    skipped_segments_path: str,
    latent_dim: int = 32,
    init_filters: int = 128,
    layers: List[int] = [2, 2, 2],
    best_model_path: str = './logs/CNNAutoencoder/best/run_01.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    device = torch.device(device)

    # Load the hexaboard data
    baseline_hexaboard = load_hexaboard(baseline_hexaboard_path)
    good_hexaboard = load_hexaboard(good_hexaboard_path)
    _, _, height, width, _ = baseline_hexaboard.shape
    
    # Load the model
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Calibrate metrics and get SSIM thresholds
    pw_metrics, ae_metrics = calibrate_metrics(
        baseline_hexaboard=baseline_hexaboard,
        good_hexaboard=good_hexaboard,
        model=model,
        device=device,
        json_map_path=json_map_path,
    )

    # Load the skipped segments
    skipped_segments = load_skipped_segments(skipped_segments_path)

    # Visualize the threshold comparison for both methods
    plot_threshold_comparison(
        pw_metrics=pw_metrics,
        ae_metrics=ae_metrics,
        skipped_segments=skipped_segments,
        save_fig='./logs/CNNAutoencoder/output/threshold_comparison.png'
    )

    # Save the results to .npy files
    np.save('./calibrations/pw_threshold.npy', pw_metrics[0])
    np.save('./calibrations/pw_bad_ssims.npy', pw_metrics[1])
    np.save('./calibrations/pw_good_ssims.npy', pw_metrics[2])
    np.save('./calibrations/ae_threshold.npy', ae_metrics[0])
    np.save('./calibrations/ae_bad_ssims.npy', ae_metrics[1])
    np.save('./calibrations/ae_good_ssims.npy', ae_metrics[2])


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    set_seed(42)

    # Run the calibration
    main(
        baseline_hexaboard_path=args.baseline_hexaboard_path,
        good_hexaboard_path=args.good_hexaboard_path,
        json_map_path=args.json_map_path,
        skipped_segments_path=args.skipped_segments_path,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers,
        best_model_path=args.best_model_path,
        device=args.device
    )