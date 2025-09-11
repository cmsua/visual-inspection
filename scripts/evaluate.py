import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models import CNNAutoencoder
from src.utils.data import HexaboardDataset
from src.utils.viz import plot_ae_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CNNAutoencoder model on hexaboard images.")

    # Data I/O arguments
    parser.add_argument('--good-hexaboard', type=str, default='./data/test', help="Folder with a good hexaboard")
    parser.add_argument('--bad-hexaboard', type=str, default='./data/bad_example', help="Folder with a bad hexaboard")
    parser.add_argument('--best-model-path', type=str, default='./logs/CNNAutoencoder/best/run_01.pt', help="Path to the best model weights")

    # Model architecture arguments
    parser.add_argument('--latent-dim', type=int, default=32, help="Bottleneck dimension")
    parser.add_argument('--init-filters', type=int, default=128, help="Initial number of filters in the model")
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 2, 2], help="Number of CNN stages and their blocks")

    # Plotting arguments
    parser.add_argument('--display-segment-idx', type=int, default=83, help="Segment index to display")

    # Dataloading/device arguments
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for training")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', action='store_true')

    return parser.parse_args()


@torch.no_grad()
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

    # Convert np.ndarray to torch.Tensor: (H, W, C) -> (C, H, W)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the hexaboard datasets
    good_dataset = HexaboardDataset(root=args.good_hexaboard, transform=transform)
    bad_dataset = HexaboardDataset(root=args.bad_hexaboard, transform=transform)

    good_dataloader = DataLoader(
        dataset=good_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    bad_dataloader = DataLoader(
        dataset=bad_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Initialize the model
    model = CNNAutoencoder(
        height=good_dataset.height,
        width=good_dataset.width,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers
    ).to(device)
    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Evaluation metrics
    good_test_loss = 0.0
    y_true_good_list = []
    y_pred_good_list = []
    bad_test_loss = 0.0
    y_true_bad_list = []
    y_pred_bad_list = []

    # Evaluate the model on the good and bad hexaboards
    for batch in good_dataloader:
        batch = batch.to(device, non_blocking=args.pin_memory)
        outputs = model(batch)
        good_test_loss += criterion(outputs, batch).item()

        y_true_good_list.append(batch.cpu().numpy())
        y_pred_good_list.append(torch.sigmoid(outputs).cpu().numpy())

    for batch in bad_dataloader:
        batch = batch.to(device, non_blocking=args.pin_memory)
        outputs = model(batch)
        bad_test_loss += criterion(outputs, batch).item()

        y_true_bad_list.append(batch.cpu().numpy())
        y_pred_bad_list.append(torch.sigmoid(outputs).cpu().numpy())

    good_test_loss /= len(good_dataloader)
    bad_test_loss /= len(bad_dataloader)
    print(f"good_test_loss: {good_test_loss}")
    print(f"bad_test_loss: {bad_test_loss}")

    y_true_good = np.concatenate(y_true_good_list, axis=0)
    y_pred_good = np.concatenate(y_pred_good_list, axis=0)
    y_true_bad = np.concatenate(y_true_bad_list, axis=0)
    y_pred_bad = np.concatenate(y_pred_bad_list, axis=0)

    # Visualize the comparison between the good and bad hexaboards
    plot_ae_comparison(
        y_true_good=y_true_good,
        y_pred_good=y_pred_good,
        y_true_bad=y_true_bad,
        y_pred_bad=y_pred_bad,
        segment_idx=args.display_segment_idx
    )


if __name__ == '__main__':
    main()