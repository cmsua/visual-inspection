import json
from pathlib import Path

import numpy as np
import pytest

import torch

from src.configs import TrainConfig
from src.engine import AutoencoderTrainer, Trainer
from src.inspection.model import load_inspection_model
from src.models import CNNAutoencoder
from src.utils.data import HexaboardDataset, load_hexaboard, load_skipped_segments
from src.utils.get_results import InspectionResults


def test_dataset_validates_every_board_and_supports_file(tmp_path: Path) -> None:
    shape = (8, 5, 10, 12, 3)
    board = np.zeros(shape, dtype=np.uint8)
    board[..., 0] = 255
    path = tmp_path / 'board.npy'
    np.save(path, board)
    dataset = HexaboardDataset(path, expected_shape=shape)
    assert len(dataset) == 32
    segment = dataset[0]
    assert segment.shape == (3, 10, 12)
    assert torch.all(segment[2] == 1)
    assert torch.all(segment[:2] == 0)
    assert load_hexaboard(str(path), expected_shape=shape)[0, 0, 0, 0].tolist() == [0, 0, 1]
    np.save(tmp_path / 'wrong.npy', np.zeros((8, 5, 11, 12, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected board shape"):
        HexaboardDataset(tmp_path, expected_shape=shape)


def test_empty_dataset_and_invalid_skip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No `.npy`"):
        HexaboardDataset(tmp_path)

    path = tmp_path / 'skip.json'
    path.write_text(json.dumps([[8, 0]]))
    with pytest.raises(ValueError, match="outside grid"):
        load_skipped_segments(str(path))


def test_results_round_trip_and_hybrid(tmp_path: Path) -> None:
    result = InspectionResults.from_segment_flags(
        (8, 5),
        iter([(0, 1), (0, 2)]),
        iter([(0, 2)]),
        {(0, 0)},
        'baseline',
        'new'
    )
    path = tmp_path / 'board.inspection.json'
    result.save_json(str(path))
    loaded = InspectionResults.load_json(str(path))
    assert loaded.pw_flags[0, 0] == -1
    assert loaded.pw_flags[0, 1] == 1
    assert loaded.hybrid_flags[0, 1] == 0
    assert loaded.hybrid_flags[0, 2] == 1


def _trainer(tmp_path: Path, interval: str, autoencoder: bool = True) -> Trainer:
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Dropout(0.1))
    data = torch.rand(4, 3, 8, 8)
    if not autoencoder:
        data = torch.utils.data.TensorDataset(data, data)

    config = TrainConfig(
        batch_size=2,
        num_epochs=2,
        criterion={'name': 'bce_with_logits_loss', 'kwargs': {}},
        scheduler={'name': 'exponential_lr', 'kwargs': {'gamma': 0.5}},
        scheduler_interval=interval,
        logging_dir=str(tmp_path),
        progress_bar=False,
        eval_strategy='steps',
        eval_steps=3,
        save_fig=False
    )
    cls = AutoencoderTrainer if autoencoder else Trainer
    return cls(model, data, data, data, device=torch.device('cpu'), config=config)


@pytest.mark.parametrize('interval,updates', [('epoch', 2), ('optimizer_step', 4), ('validation', 3)])
def test_autoencoder_inherits_scheduler_and_validation(tmp_path: Path, interval: str, updates: int) -> None:
    trainer = _trainer(tmp_path, interval)
    learning_rate = trainer.optimizer.param_groups[0]['lr']
    trainer.train()
    assert trainer.history['step'] == [2, 3, 4]
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(learning_rate * 0.5**updates)
    assert trainer.model.training
    restored = _trainer(tmp_path / 'restored', interval)
    restored.load_checkpoint(trainer.checkpoint_path)
    assert restored.start_epoch == 2
    assert restored.optimizer.param_groups[0]['lr'] == trainer.optimizer.param_groups[0]['lr']
    assert len(restored.evaluate()) == 3


def test_generic_trainer_keeps_supervised_dispatch(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, 'epoch', autoencoder=False)
    trainer.train()
    assert len(trainer.evaluate('bce')) == 4


def test_autoencoder_graceful_stop_saves_completed_update(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, 'optimizer_step')
    original = trainer.optimizer.step

    def request_stop(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        trainer._stop_requested = True
        return result

    trainer.optimizer.step = request_stop
    trainer.train()
    checkpoint = torch.load(trainer.checkpoint_path, weights_only=True)
    assert checkpoint['global_step'] == 1
    assert checkpoint['scheduler_state_dict']['last_epoch'] == 1


def test_checkpoint_architecture_is_inferred(tmp_path: Path) -> None:
    model = CNNAutoencoder(latent_dim=64, init_filters=8)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    loaded = load_inspection_model(str(path), 1060, 1882, torch.device('cpu'))
    assert loaded.latent_dim == 64
    with pytest.raises(ValueError, match="does not match"):
        load_inspection_model(str(path), 1060, 1882, torch.device('cpu'), {'latent_dim': 32})
