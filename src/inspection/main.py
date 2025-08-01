import argparse

import numpy as np

import torch

from ..inferences import autoencoder_inference, pixelwise_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Visual Inspection Procedure",
        description="Inspects hexaboard segments for defects",
        epilog="University of Alabama"
    )

    # Data I/O arguments
    parser.add_argument('-b', '--baseline-images-path', type=str, required=True, help="Path to the baseline hexaboard images")
    parser.add_argument('-n', '--new-images-path', type=str, required=True, help="Path to the new hexaboard images to inspect")
    parser.add_argument('-w', '--best-model-path', type=str, default='./logs/ResNetAutoencoder/best/run_01.pt', help="Path to the best model weights")

    # Model architecture arguments
    parser.add_argument('--latent-dim', type=int, default=128, help="Bottleneck dimension")
    parser.add_argument('--init-filters', type=int, default=64, help="Initial number of filters in the model")
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 2, 2], help="Number of ResNet layers and their BasicBlocks")

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for training")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    # Load the hexaboard data
    baseline_hexaboard = np.load(args.baseline_images_path)
    new_hexaboard = np.load(args.new_images_path)
    
    # Ensure they are 5D arrays
    if baseline_hexaboard.ndim != 5 or new_hexaboard.ndim != 5:
        raise ValueError("Both hexaboard files must contain 5D arrays (H_seg, V_seg, height, width, num_channels)")

    # Perform inferences
    threshold = 0.5  # adjust as needed
    ae_indices = autoencoder_inference(
        hexaboard=new_hexaboard,
        threshold=threshold,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers,
        best_model_path=args.best_model_path,
        device=device
    )
    pw_indices = pixelwise_inference(
        baseline_hexaboard=baseline_hexaboard,
        new_hexaboard=new_hexaboard,
        threshold=threshold
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