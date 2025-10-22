import argparse
from typing import List

import numpy as np

import torch

from src.inferences import autoencoder_inference, pixelwise_inference
from src.models import CNNAutoencoder
from src.utils import InspectionResults, set_seed
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


def main(
    baseline_hexaboard_path: str,
    new_hexaboard_path: str,
    skipped_segments_path: str,
    ae_threshold_path: str = './calibrations/ae_threshold.npy',
    pw_threshold_path: str = './calibrations/pw_threshold.npy',
    latent_dim: int = 32,
    init_filters: int = 128,
    layers: List = [2, 2, 2],
    best_model_path: str = './logs/CNNAutoencoder/best/run_01.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> InspectionResults:
    device = torch.device(device)

    # Reproducibility settings
    set_seed(42)

    # Load the hexaboard data
    baseline_hexaboard = load_hexaboard(baseline_hexaboard_path)
    new_hexaboard = load_hexaboard(new_hexaboard_path)
    H_seg, V_seg, height, width, channels = baseline_hexaboard.shape
    segment_shape = (height, width, channels)
    grid_shape = (H_seg, V_seg)

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

    # Load the thresholds
    ae_threshold = np.load(ae_threshold_path)
    pw_threshold = np.load(pw_threshold_path)

    # Load the skipped segments
    skipped_segments = load_skipped_segments(skipped_segments_path)

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

    # Compile the inspection results
    results = InspectionResults.from_segment_flags(
        shape=grid_shape,
        pixel_flagged=pw_indices,
        autoencoder_flagged=ae_indices,
        skipped_segments=skipped_segments,
        baseline_path=baseline_hexaboard_path,
        inspected_path=new_hexaboard_path,
        segment_shape=segment_shape
    )

    return results


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Run the inspection
    inspection_results = main(
        baseline_hexaboard_path=args.baseline_hexaboard_path,
        new_hexaboard_path=args.new_hexaboard_path,
        skipped_segments_path=args.skipped_segments_path,
        ae_threshold_path=args.ae_threshold_path,
        pw_threshold_path=args.pw_threshold_path,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers,
        best_model_path=args.best_model_path,
        device=args.device
    )

    flags = inspection_results.metadata.get('flagged_segments', {})
    print(f"Pixel-only flagged segments: {flags.get('pixel_only', [])}")
    print(f"Autoencoder-only flagged segments: {flags.get('autoencoder_only', [])}")
    print(f"Flagged by both methods: {flags.get('both_methods', [])}")

    summary = inspection_results.summary()
    print(
        "Inspection summary - "
        f"pixel: {summary['pixel_flagged']} | "
        f"autoencoder: {summary['autoencoder_flagged']} | "
        f"hybrid: {summary['hybrid_flagged']} | "
        f"skipped: {summary['skipped']}"
    )