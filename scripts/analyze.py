import os
import json
import argparse
from typing import List

import numpy as np

import torch

from src.inferences import autoencoder_inference, pixelwise_inference
from src.models import CNNAutoencoder
from src.utils import set_seed, agg_confusion_matrix
from src.utils.data import load_hexaboard, load_skipped_segments
from src.utils.viz import plot_confusion_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a CNNAutoencoder model on hexaboard images.")

    # Data loading arguments
    parser.add_argument('--train-data-dir', type=str, default='./data/train', help="Train data folder")
    parser.add_argument('--val-data-dir', type=str, default='./data/val', help="Validation data folder")
    parser.add_argument('--test-data-dir', type=str, default='./data/test', help="Test data folder")
    parser.add_argument('--bad-data-dir', type=str, default='./data/bad', help="Folder with bad hexaboards")
    parser.add_argument('-j', '--json-map-path', type=str, default='./calibrations/damaged_segments.json', help="Path to the JSON map file")
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


@torch.no_grad()
def main(
    train_data_dir: str = './data/train',
    val_data_dir: str = './data/val',
    test_data_dir: str = './data/test',
    bad_data_dir: str = './data/bad',
    json_map_path: str = './calibrations/damaged_segments.json',
    skipped_segments_path: str = './calibrations/skipped_segments.json',
    ae_threshold_path: str = './calibrations/ae_threshold.npy',
    pw_threshold_path: str = './calibrations/pw_threshold.npy',
    latent_dim: int = 32,
    init_filters: int = 128,
    layers: List = [2, 2, 2],
    best_model_path: str = './logs/CNNAutoencoder/best/run_01.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    device = torch.device(device)

    # Load the baseline hexaboard to get image dimensions
    baseline_hexaboard = load_hexaboard(os.path.join(train_data_dir, 'aligned_images1.npy'))
    H_seg, V_seg, height, width, _ = baseline_hexaboard.shape

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

    # Build a list of paths for all good hexaboards
    good_hexaboard_paths = []
    for data_dir in [train_data_dir, val_data_dir, test_data_dir]:
        for filename in os.listdir(data_dir):
            if filename != 'aligned_images1.npy':
                good_hexaboard_paths.append(os.path.join(data_dir, filename))

    # List of paths for all bad hexaboards
    bad_hexaboard_paths = sorted(os.listdir(bad_data_dir))

    # Build a mapping from bad filename basename -> list of true bad coords
    true_bad_map = {}
    with open(json_map_path, 'r') as f:
        bad_segments_mapping = json.load(f)['files']
        
    for key, val in bad_segments_mapping.items():
        base = os.path.basename(key)
        coords = []
        for coord in val.get('damaged', []):
            coords.append((coord['row'], coord['col']))

        true_bad_map[base] = coords

    # Derive extended skipped segments
    bad_coords = {coord for coords in true_bad_map.values() for coord in coords}
    all_coords = {(h, v) for h in range(H_seg) for v in range(V_seg)}
    extended_skipped_segments = set(skipped_segments) | (all_coords - bad_coords)

    # Analyze the good hexaboards
    ae_pred_good = []
    pw_pred_good = []
    double_pred_good = []
    for hexaboard_path in good_hexaboard_paths:
        hexaboard = load_hexaboard(hexaboard_path)

        ae_indices = autoencoder_inference(
            hexaboard=hexaboard,
            threshold=ae_threshold,
            model=model,
            device=device,
            skipped_segments=extended_skipped_segments
        )
        pw_indices = pixelwise_inference(
            baseline_hexaboard=baseline_hexaboard,
            new_hexaboard=hexaboard,
            threshold=pw_threshold,
            skipped_segments=extended_skipped_segments
        )
        double_flagged_indices = sorted(list(set(ae_indices) & set(pw_indices)))

        ae_pred_good.append(ae_indices)
        pw_pred_good.append(pw_indices)
        double_pred_good.append(double_flagged_indices)

    # Analyze the bad hexaboards
    ae_pred_bad = []
    pw_pred_bad = []
    double_pred_bad = []
    for filename in bad_hexaboard_paths:
        hexaboard_path = os.path.join(bad_data_dir, filename)
        hexaboard = load_hexaboard(hexaboard_path)

        ae_indices = autoencoder_inference(
            hexaboard=hexaboard,
            threshold=ae_threshold,
            model=model,
            device=device,
            skipped_segments=extended_skipped_segments
        )
        pw_indices = pixelwise_inference(
            baseline_hexaboard=baseline_hexaboard,
            new_hexaboard=hexaboard,
            threshold=pw_threshold,
            skipped_segments=extended_skipped_segments
        )
        double_flagged_indices = sorted(list(set(ae_indices) & set(pw_indices)))

        ae_pred_bad.append(ae_indices)
        pw_pred_bad.append(pw_indices)
        double_pred_bad.append(double_flagged_indices)
        
    # Total segments per board (exclude skipped segments)
    total_segments_per_board = H_seg * V_seg - len(extended_skipped_segments)
    ae_cm = agg_confusion_matrix(
        bad_hexaboard_paths=bad_hexaboard_paths,
        pred_good_list=ae_pred_good,
        pred_bad_list=ae_pred_bad,
        true_bad_map=true_bad_map,
        total_per_board=total_segments_per_board
    )
    pw_cm = agg_confusion_matrix(
        bad_hexaboard_paths=bad_hexaboard_paths,
        pred_good_list=pw_pred_good,
        pred_bad_list=pw_pred_bad,
        true_bad_map=true_bad_map,
        total_per_board=total_segments_per_board
    )
    double_cm = agg_confusion_matrix(
        bad_hexaboard_paths=bad_hexaboard_paths,
        pred_good_list=double_pred_good,
        pred_bad_list=double_pred_bad,
        true_bad_map=true_bad_map,
        total_per_board=total_segments_per_board
    )

    # Plot the three confusion matrices
    plot_confusion_matrices(
        ae_cm=ae_cm,
        pw_cm=pw_cm,
        double_cm=double_cm,
        title_prefix="Aggregated over all boards",
        save_fig='./logs/CNNAutoencoder/output/confusion_matrices.png'
    )


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    set_seed(42)

    # Analyze the model's performance on good and bad hexaboard images
    main(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        test_data_dir=args.test_data_dir,
        bad_data_dir=args.bad_data_dir,
        json_map_path=args.json_map_path,
        skipped_segments_path=args.skipped_segments_path,
        ae_threshold_path=args.ae_threshold_path,
        pw_threshold_path=args.pw_threshold_path,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers,
        best_model_path=args.best_model_path,
        device=args.device
    )