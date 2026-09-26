import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import yaml

import torch
import torch.multiprocessing as mp

from src.configs import AutoencoderConfig, TrainConfig
from src.engine import AutoencoderTrainer
from src.models import CNNAutoencoder
from src.utils import cleanup_ddp, set_seed, setup_ddp
from src.utils.data import HexaboardDataset
from src.utils.viz import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNNAutoencoder model on hexaboard images.")

    # Model and configurations arguments
    parser.add_argument(
        '--config-path',
        type=str,
        default='./configs/train_CNNAutoencoder.yaml',
        help="Path to YAML config"
    )
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Checkpoint to restore trainer state")

    # Data loading arguments
    parser.add_argument('--train-data-dir', type=str, default='./data/train', help="Train data folder")
    parser.add_argument('--val-data-dir', type=str, default='./data/val', help="Validation data folder")

    return parser.parse_args()


def main(
    rank: int,
    world_size: int,
    config_path: str,
    checkpoint_path: Optional[str] = None,
    train_data_dir: str = './data/train',
    val_data_dir: str = './data/val'
) -> None:
    # Reproducibility settings
    set_seed(42)

    # Load the YAML file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = AutoencoderConfig.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize multi-GPU processing
    setup_ddp(rank, world_size)
    try:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Create the hexaboard datasets
        train_dataset = HexaboardDataset(root=train_data_dir)
        val_dataset = HexaboardDataset(root=val_data_dir)

        # Initialize the model
        model = CNNAutoencoder(config=model_config).to(device)

        # Initialize the trainer
        trainer = AutoencoderTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=train_config
        )

        # Resume from checkpoint if provided
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)

        # Train the model
        history, model = trainer.train()

    finally:
        cleanup_ddp()

    # Save the training history plot
    output_path = os.path.join(trainer.outputs_dir, f'{trainer.run_name}.png') if train_config.save_fig else None
    if rank == 0:
        figure = plot_history(history, savefig=output_path, show=False)
        plt.close(figure)


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Multi-GPU processing
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(world_size, args.config_path, args.checkpoint_path, args.train_data_dir, args.val_data_dir),
            nprocs=world_size
        )
    else:
        # 1 GPU or CPU: run the same code on rank 0
        main(
            rank=0,
            world_size=1,
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            train_data_dir=args.train_data_dir,
            val_data_dir=args.val_data_dir
        )
