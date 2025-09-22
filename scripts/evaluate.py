import os
import yaml
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.configs import CNNAutoencoderConfig, TrainConfig
from src.loss import LOSS_REGISTRY
from src.models import CNNAutoencoder
from src.utils import get_loss_from_config, set_seed
from src.utils.data import HexaboardDataset
from src.utils.viz import plot_ae_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CNNAutoencoder model on hexaboard images.")

    # Model and configurations arguments
    parser.add_argument('--config-path', type=str, default='./configs/train_CNNAutoencoder.yaml', help="Path to YAML config")
    parser.add_argument('--best-model-path', type=str, default='./logs/CNNAutoencoder/best/run_01.pt', help="Path to the best model weights")

    # Data loading arguments
    parser.add_argument('--good-hexaboard-dir', type=str, default='./data/test', help="Folder with a good hexaboard")
    parser.add_argument('--bad-hexaboard-dir', type=str, default='./data/bad_example', help="Folder with a bad hexaboard")

    # Plotting arguments
    parser.add_argument('--display-segment-idx', type=int, default=83, help="Segment index to display")

    return parser.parse_args()


@torch.no_grad()
def main(
    config_path: str,
    best_model_path: str = None,
    good_hexaboard_dir: str = './data/test',
    bad_hexaboard_dir: str = './data/bad_example',
    display_segment_idx: int = 83
):
    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = CNNAutoencoderConfig.from_dict(config['model'])
    eval_config = TrainConfig.from_dict(config['train'])

    # Convert np.ndarray to torch.Tensor: (H, W, C) -> (C, H, W)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the hexaboard datasets
    good_dataset = HexaboardDataset(root=good_hexaboard_dir, transform=transform)
    bad_dataset = HexaboardDataset(root=bad_hexaboard_dir, transform=transform)

    good_dataloader = DataLoader(
        dataset=good_dataset,
        batch_size=eval_config.batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
        pin_memory=eval_config.pin_memory
    )
    bad_dataloader = DataLoader(
        dataset=bad_dataset,
        batch_size=eval_config.batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
        pin_memory=eval_config.pin_memory
    )

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNAutoencoder(config=model_config).to(device)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Loss function
    try:
        criterion = get_loss_from_config(eval_config.criterion, LOSS_REGISTRY)
    except:
        raise ValueError("Missing criterion configuration.")

    # Evaluation metrics
    good_test_loss = 0.0
    y_true_good_list = []
    y_pred_good_list = []
    
    bad_test_loss = 0.0
    y_true_bad_list = []
    y_pred_bad_list = []

    # Evaluate the model on the good and bad hexaboards
    for batch in good_dataloader:
        batch = batch.to(device, non_blocking=eval_config.pin_memory)
        outputs = model(batch)
        good_test_loss += criterion(outputs, batch).item()

        y_true_good_list.append(batch.cpu().numpy())
        y_pred_good_list.append(torch.sigmoid(outputs).cpu().numpy())

    for batch in bad_dataloader:
        batch = batch.to(device, non_blocking=eval_config.pin_memory)
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

    # Visualize the comparison between the good and bad hexaboards and save the figure if specified
    output_path = os.path.join(
        eval_config.logging_dir,
        model.__class__.__name__,
        'outputs',
        'ae_performance.png'
    ) if eval_config.save_fig else None
    plot_ae_comparison(
        y_true_good=y_true_good,
        y_pred_good=y_pred_good,
        y_true_bad=y_true_bad,
        y_pred_bad=y_pred_bad,
        segment_idx=display_segment_idx,
        save_fig=output_path
    )


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    set_seed(42)

    # Evaluate the model on a good and a bad hexaboard
    main(
        config_path=args.config_path,
        best_model_path=args.best_model_path,
        good_hexaboard_dir=args.good_hexaboard_dir,
        bad_hexaboard_dir=args.bad_hexaboard_dir,
        display_segment_idx=args.display_segment_idx
    )