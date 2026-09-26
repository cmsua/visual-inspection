import json
from pathlib import Path

import numpy as np
import pytest

import torch

from src.configs.inspection_config import InspectionConfig
from src.inspection.artifacts import file_signatures
from src.inspection.run import inspect_board, validate_calibration
from src.utils.get_results import InspectionResults
from src.utils.get_threshold import calibrate_metrics


def test_calibration_scores_each_normal_coordinate_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (8, 5, 9, 9, 3)
    paths = []
    for name, value in [('good', 10), ('bad1', 80), ('bad2', 90)]:
        path = tmp_path / f'{name}.npy'
        np.save(path, np.full(shape, value, dtype=np.uint8))
        paths.append(str(path))

    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'files': {path: {'damaged': [{'row': 1, 'col': 1}]} for path in paths[1:]}}))
    calls = []

    def scores(baseline: np.ndarray, board: np.ndarray, coordinates: list, model: object, device: object) -> object:
        for row, col in coordinates:
            calls.append((int(board[0, 0, 0, 0, 0]), row, col))
            value = float(board[0, 0, 0, 0, 0]) / 255
            yield row, col, value, value

    monkeypatch.setattr('src.utils.get_threshold.segment_scores', scores)
    skipped = {(r, c) for r in range(8) for c in range(5)} - {(1, 1), (1, 2)}
    pixel, ae = calibrate_metrics(
        np.zeros(shape, dtype=np.uint8),
        paths[:1],
        torch.nn.Identity(),
        torch.device('cpu'),
        str(labels),
        bad_hexaboard_paths=paths[1:],
        skipped_segments=skipped,
        baseline_is_bgr=True
    )
    assert len(calls) == 4
    assert calls.count((10, 1, 1)) == 1
    assert pixel[0][1, 1] == pytest.approx(80 / 255)
    assert pixel[0][1, 2] > 10 / 255
    assert np.isfinite(ae[0]).all()


def test_calibration_rejects_stale_model_and_thresholds(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ('baseline.npy', 'model.pt', 'skipped.json')]
    for path in paths:
        path.write_bytes(b'original')

    for name in ('ae_threshold.npy', 'pw_threshold.npy'):
        np.save(tmp_path / name, np.zeros((8, 5)))

    config = InspectionConfig(
        baseline_path=str(paths[0]),
        checkpoint_path=str(paths[1]),
        skipped_segments_path=str(paths[2]),
        calibration_dir=str(tmp_path)
    ).resolve()
    manifest = {
        'dependencies': file_signatures(map(str, paths)),
        'threshold_files': file_signatures([str(tmp_path / name) for name in ('ae_threshold.npy', 'pw_threshold.npy')])
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    validate_calibration(config)
    paths[1].write_bytes(b'changed model')
    with pytest.raises(ValueError, match='changed'):
        validate_calibration(config)


def test_persisted_inspection_reloads_without_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    board = tmp_path / 'board.npy'
    board.write_bytes(b'board')
    config = InspectionConfig(board_path=str(board)).resolve()
    monkeypatch.setattr('src.inspection.run.validate_calibration', lambda config: None)
    monkeypatch.setattr(
        'src.inspection.run.inspection_signature',
        lambda config, path: {'files': file_signatures([path])}
    )
    calls = []

    def infer(**kwargs: object) -> InspectionResults:
        calls.append(kwargs)
        return InspectionResults.from_segment_flags((8, 5), [(1, 1)], [], {(0, 0)}, 'baseline', str(board))

    monkeypatch.setattr('src.inspection.run.run_inspection', infer)
    inspect_board(config)
    inspect_board(config)
    assert len(calls) == 1
    board.write_bytes(b'new board')
    inspect_board(config)
    assert len(calls) == 2
