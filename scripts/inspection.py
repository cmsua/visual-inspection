import argparse

import numpy as np

import torch

from src.inferences import autoencoder_inference, pixelwise_inference
from src.models import CNNAutoencoder
from src.utils import set_seed
from src.utils.data import load_hexaboard, load_skipped_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Visual Inspection Procedure",
        description="Inspects hexaboard segments for defects",
        epilog="University of Alabama"
    )

    # Data loading arguments
    parser.add_argument('-b', '--baseline-hexaboard-path', type=str, required=True, help="Path to the baseline hexaboard images")
    parser.add_argument('-n', '--new-hexaboard-path', type=str, required=True, help="Path to the new hexaboard images to inspect")
    parser.add_argument('-s', '--skipped-segments-path', type=str, default='./calibrations/skipped_segments.json', help="Path to the JSON file containing the list of segments to skip")

    # Threshold arguments
    parser.add_argument('--ae-threshold-path', type=str, default='./calibrations/ae_threshold.npy', help="Path to the autoencoder threshold .npy file")
    parser.add_argument('--pw-threshold-path', type=str, default='./calibrations/pw_threshold.npy', help="Path to the pixel-wise threshold .npy file")

    # Model architecture arguments
    parser.add_argument('--latent-dim', type=int, default=32, help="Bottleneck dimension")
    parser.add_argument('--init-filters', type=int, default=128, help="Initial number of filters in the model")
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 2, 2], help="Number of CNN stages and their blocks")
    parser.add_argument('-w', '--best-model-path', type=str, default='./logs/CNNAutoencoder/best/run_01.pt', help="Path to the best model weights")

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for inference")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    set_seed(42)
    device = torch.device(args.device)

    # Load the hexaboard data
    baseline_hexaboard = load_hexaboard(args.baseline_hexaboard_path)
    new_hexaboard = load_hexaboard(args.new_hexaboard_path)

    _, _, height, width, _ = baseline_hexaboard.shape

    # Load the model
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers
    ).to(device)
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    model.eval()

    # Load the thresholds
    ae_threshold = np.load(args.ae_threshold_path)
    pw_threshold = np.load(args.pw_threshold_path)

    # Load the skipped segments
    skipped_segments = load_skipped_segments(args.skipped_segments_path)

    # Perform inferences
    ae_indices = autoencoder_inference(
        hexaboard=new_hexaboard,
        threshold=ae_threshold,
        model=model,
        device=device,
        skipped_segments=skipped_segments
    )
    pw_indices = pixelwise_inference(
        baseline_hexaboard=baseline_hexaboard,
        new_hexaboard=new_hexaboard,
        threshold=pw_threshold,
        skipped_segments=skipped_segments
    )

    # Lists of flagged segments
    double_flagged = sorted(list(set(pw_indices) & set(ae_indices)))
    ml_flagged = sorted(set(ae_indices) - set(double_flagged))
    pixel_flagged = sorted(set(pw_indices) - set(double_flagged))
    all_flagged = sorted(list(set(ae_indices).union(pw_indices)))

    print(f"Double flagged segments: {double_flagged}")
    print(f"Autoencoder flagged segments: {ml_flagged}")
    print(f"Pixel-wise flagged segments: {pixel_flagged}")
    print(f"All flagged segments: {all_flagged}")


if __name__ == '__main__':
    main()