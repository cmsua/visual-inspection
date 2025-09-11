import os
import argparse

import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms

from src.configs import TrainConfig
from src.engine import AutoencoderTrainer
from src.models import CNNAutoencoder
from src.utils import EarlyStopping
from src.utils.data import HexaboardDataset
from src.utils.viz import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNNAutoencoder model on hexaboard images.")

    # Data I/O arguments
    parser.add_argument('--train-dataset', type=str, default='./data/train', help="Train data folder")
    parser.add_argument('--val-dataset', type=str, default='./data/val', help="Validation data folder")
    parser.add_argument('--log-dir', type=str, default='./logs', help="Model weights & checkpoints directory")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to an existing checkpoint to resume from")

    # Model architecture arguments
    parser.add_argument('--latent-dim', type=int, default=32, help="Bottleneck dimension")
    parser.add_argument('--init-filters', type=int, default=128, help="Initial number of filters in the model")
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 2, 2], help="Number of CNN stages and their blocks")

    # Training hyperparameters arguments
    parser.add_argument('--train-val-test-split', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--exponential-lr-gamma', type=float, default=0.96)
    parser.add_argument('--early-stopping-patience', type=int, default=5)

    # Logging/plotting arguments
    parser.add_argument('--logging-steps', type=int, default=25, help="Steps between logging")
    parser.add_argument('--plot-history', action='store_true', help="Plot training history after training")

    # Dataloading/device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for training")
    parser.add_argument('--num-workers', type=int, default=0)

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

    # Convert np.ndarray to torch.Tensor: (H, W, C) -> (C, H, W)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the hexaboard datasets
    train_dataset = HexaboardDataset(root=args.train_dataset, transform=transform)
    val_dataset = HexaboardDataset(root=args.val_dataset, transform=transform)

    # Initialize the model
    model = CNNAutoencoder(
        height=train_dataset.height,
        width=train_dataset.width,
        latent_dim=args.latent_dim,
        init_filters=args.init_filters,
        layers=args.layers
    ).to(device)
    
    # Training configurations
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        start_epoch=0,
        logging_dir=args.log_dir,
        logging_steps=args.logging_steps,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    # Initialize the trainer
    learning_rate = args.learning_rate
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exponential_lr_gamma)
    callbacks = EarlyStopping(monitor='val_loss', mode='min', patience=args.early_stopping_patience)
    trainer = AutoencoderTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=[callbacks],
        config=train_config
    )

    # Resume from checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        try:
            trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Train the model
    history, model = trainer.train()

    # Visualize training history (optional)
    if args.plot_history:
        plot_history(history)


if __name__ == '__main__':
    main()