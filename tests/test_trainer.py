import shutil
import tempfile
from typing import Tuple

import pytest

import torch
from torchvision import transforms

from src.configs import TrainConfig
from src.engine import AutoencoderTrainer
from src.models import CNNAutoencoder
from src.utils import set_seed
from src.utils.data import HexaboardDataset

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = './data'


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def make_datasets() -> Tuple[HexaboardDataset, ...]:
    # Convert np.ndarray to torch.Tensor: (H, W, C) -> (C, H, W)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Use folders with less sample for speed
    train_dataset = HexaboardDataset(root=DATA_DIR + '/val', transform=transform)
    val_dataset = HexaboardDataset(root=DATA_DIR + '/test', transform=transform)
    test_dataset = HexaboardDataset(root=DATA_DIR + '/bad_example', transform=transform)

    return train_dataset, val_dataset, test_dataset


def make_trainer(temp_dir: str) -> AutoencoderTrainer:
    # Create datasets
    train_dataset, val_dataset, test_dataset = make_datasets()

    # Initialize the model
    model = CNNAutoencoder(
        height=1016,
        width=1640,
        latent_dim=64,
        init_filters=128,
        layers=[2, 2, 2]
    ).to(device)

    # Training configurations
    train_config = TrainConfig(
        batch_size=8,
        criterion={
            'name': 'bce_with_logits_loss',
            'kwargs': {
                'reduction': 'mean'
            }
        },
        optimizer={
            'name': 'adam',
            'kwargs': {
                'lr': 1e-3,
                'weight_decay': 1e-4
            }
        },
        num_epochs=1,
        logging_dir=temp_dir,
        logging_steps=25,
        progress_bar=True,
        save_best=True,
        save_ckpt=True,
        save_fig=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize the trainer
    trainer = AutoencoderTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        config=train_config
    )

    return trainer


def test_training_step(temp_dir: str) -> None:
    trainer = make_trainer(temp_dir)
    trainer.train()

    # Check if the model was trained
    assert trainer.model is not None, "Model should be trained and not None."
    assert trainer.num_epochs > 0, "Training should progress beyond the initial epoch."


def test_load_checkpoint(temp_dir: str) -> None:
    trainer = make_trainer(temp_dir)
    trainer.train()

    # Simulate saving a checkpoint
    trainer.save_checkpoint(1)

    # Create a new trainer instance and load the checkpoint
    new_trainer = make_trainer(temp_dir)
    new_trainer.load_checkpoint(trainer.checkpoint_path)

    # Check if the model weights are the same
    saved_weights = trainer.model.state_dict()
    new_weights = new_trainer.model.state_dict()
    
    assert saved_weights.keys() == new_weights.keys(), "Model state dict keys should match."
    assert all(torch.equal(saved_weights[k], new_weights[k]) for k in saved_weights.keys()), "Model weights should be equal after loading checkpoint."


def test_evaluate_model(temp_dir: str) -> None:
    trainer = make_trainer(temp_dir)
    trainer.train()

    # Evaluate the model
    test_loss, y_true, y_pred = trainer.evaluate()

    # Check if the evaluation returns valid results
    assert test_loss >= 0, "Test loss should be non-negative."
    assert len(y_true) > 0 and len(y_pred) > 0, "Evaluation should return true and predicted values."
    assert len(y_true) == len(y_pred), "True and predicted values should have the same length."