import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models import CNNAutoencoder
from src.utils.data import HexaboardDataset
from src.utils.viz import plot_reconstructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CNNAutoencoder model on hexaboard images.")

    # Data I/O arguments
    parser.add_argument('--test-dataset', type=str, default='./data/test', help="Test data folder")
    parser.add_argument('--best-model-path', type=str, default='./logs/CNNAutoencoder/best/run_01.pt', help="Path to the best model weights")

    # Model architecture arguments
    parser.add_argument('--latent-dim', type=int, default=16, help="Bottleneck dimension")
    parser.add_argument('--init-filters', type=int, default=32, help="Initial number of filters in the model")
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 2, 2], help="Number of CNN stages and their blocks")

    # Plotting arguments
    parser.add_argument('--num-images', type=int, default=8, help="Number of images to visualize in the output")
    parser.add_argument('--no-plot', action='store_true', help="Disable plotting of reconstructed images")

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

    # Create the hexaboard dataset
    dataset = HexaboardDataset(root=args.test_dataset, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Initialize the model
    model = CNNAutoencoder(
        height=dataset.height,
        width=dataset.width,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers
    ).to(device)
    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0
    y_true_list = []
    y_pred_list = []

    # Evaluate the model
    for batch in dataloader:
        batch = batch.to(device, non_blocking=args.pin_memory)
        outputs = model(batch)
        test_loss += criterion(outputs, batch).item()

        y_true_list.append(batch.cpu().numpy())
        y_pred_list.append(torch.sigmoid(outputs).cpu().numpy())

    test_loss /= len(dataloader)

    print(f"test_loss: {test_loss}")

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    if not args.no_plot:
        # Visualize the original vs. reconstructed images
        plot_reconstructions(y_true, y_pred, num_images=args.num_images)


if __name__ == '__main__':
    main()