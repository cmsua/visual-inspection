import shutil
from pathlib import Path
from typing import Any, Tuple
from uuid import uuid4

import pytest

import torch

from src.configs import TrainConfig
from src.engine import AutoencoderTrainer
from src.utils import set_seed
from src.utils.data import HexaboardDataset

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = './data'
TEST_TMP_DIR = Path('./logs/test_tmp')


@pytest.fixture
def temp_dir() -> Any:
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    d = TEST_TMP_DIR / f'tmp_{uuid4().hex}'
    d.mkdir(parents=True, exist_ok=False)
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


def make_datasets() -> Tuple[HexaboardDataset, ...]:
    return torch.rand(8, 3, 12, 12), torch.rand(4, 3, 12, 12), torch.rand(4, 3, 12, 12)


def make_trainer(
    temp_dir: str,
    *,
    num_epochs: int = 1,
    eval_strategy: str = 'epoch',
    eval_steps: int = 0
) -> AutoencoderTrainer:
    # Create datasets
    train_dataset, val_dataset, test_dataset = make_datasets()

    # Initialize the model
    model = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)

    # Training configurations
    train_config = TrainConfig(
        batch_size=2,
        criterion={'name': 'bce_with_logits_loss', 'kwargs': {'reduction': 'mean'}},
        optimizer={'name': 'adam', 'kwargs': {'lr': 1e-3, 'weight_decay': 1e-4}},
        scheduler={'name': 'exponential_lr', 'kwargs': {'gamma': 0.96}},
        callbacks=[],
        num_epochs=num_epochs,
        logging_dir=temp_dir,
        logging_steps=10,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        progress_bar=False,
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
    assert all(torch.equal(saved_weights[k], new_weights[k]) for k in saved_weights.keys()), (
        "Model weights should be equal after loading checkpoint."
    )


def test_evaluate_model(temp_dir: str) -> None:
    trainer = make_trainer(temp_dir)
    trainer.train()

    # Evaluate the model
    test_loss, y_true, y_pred = trainer.evaluate()

    # Check if the evaluation returns valid results
    assert test_loss >= 0, "Test loss should be non-negative."
    assert len(y_true) > 0 and len(y_pred) > 0, "Evaluation should return true and predicted values."
    assert len(y_true) == len(y_pred), "True and predicted values should have the same length."


def test_step_eval_strategy_matches_base_epoch_boundary_validation(temp_dir: str) -> None:
    trainer = make_trainer(temp_dir, num_epochs=2, eval_strategy='steps', eval_steps=3)

    trainer.train()

    assert trainer.history['step'] == [3, 4, 6, 8]
