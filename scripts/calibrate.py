import os
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

    # Data loading arguments
    parser.add_argument('--train-data-dir', type=str, default='./data/train', help="Train data folder")
    parser.add_argument('--val-data-dir', type=str, default='./data/val', help="Validation data folder")
    parser.add_argument('--test-data-dir', type=str, default='./data/test', help="Test data folder")
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


@torch.no_grad()
def main(
    train_data_dir: str = './data/train',
    val_data_dir: str = './data/val',
    test_data_dir: str = './data/test',
    json_map_path: str = './calibrations/damaged_segments.json',
    skipped_segments_path: str = './calibrations/skipped_segments.json',
    latent_dim: int = 32,
    init_filters: int = 128,
    layers: List[int] = [2, 2, 2],
    best_model_path: str = './logs/CNNAutoencoder/best/run_01.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    device = torch.device(device)

    # Reproducibility settings
    set_seed(42)

    # Load the baseline hexaboard to get image dimensions
    baseline_hexaboard = load_hexaboard(os.path.join(train_data_dir, 'aligned_images1.npy'))
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

    # Build a list of full paths for all good hexaboards
    good_hexaboard_paths = []
    for data_dir in [train_data_dir, val_data_dir, test_data_dir]:
        for filename in os.listdir(data_dir):
            if filename != 'aligned_images1.npy':
                good_hexaboard_paths.append(os.path.join(data_dir, filename))

    # Calibrate metrics and get thresholds
    pw_metrics, ae_metrics = calibrate_metrics(
        baseline_hexaboard=baseline_hexaboard,
        good_hexaboard_paths=good_hexaboard_paths,
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
    np.save('./calibrations/ae_bad_maes.npy', ae_metrics[1])
    np.save('./calibrations/ae_good_maes.npy', ae_metrics[2])


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Run the calibration
    main(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        test_data_dir=args.test_data_dir,
        json_map_path=args.json_map_path,
        skipped_segments_path=args.skipped_segments_path,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers,
        best_model_path=args.best_model_path,
        device=args.device
    )