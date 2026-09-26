import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import torch

from .artifacts import file_signatures
from .model import load_inspection_model
from ..configs.inspection_config import InspectionConfig
from ..utils.data import load_skipped_segments
from ..utils.data.board_contract import open_board
from ..utils.get_results import InspectionResults
from ..utils.inspection_scores import segment_scores


def _threshold_paths(directory: str) -> List[str]:
    return [str(Path(directory) / name) for name in ('ae_threshold.npy', 'pw_threshold.npy')]


def inspection_signature(config: InspectionConfig, board_path: str) -> Dict:
    paths = [config.baseline_path, board_path, config.checkpoint_path, config.skipped_segments_path]
    paths.extend(_threshold_paths(config.calibration_dir))
    manifest = Path(config.calibration_dir) / 'manifest.json'
    if manifest.exists():
        paths.append(str(manifest))

    return {'files': file_signatures(paths), 'model': config.model, 'version': 1}


def validate_calibration(config: InspectionConfig) -> None:
    manifest_path = Path(config.calibration_dir) / 'manifest.json'
    if not manifest_path.exists():
        raise ValueError(
            "Calibration provenance is missing. Run `python -m scripts.calibrate --config-path configs/inspection.yaml` first."
        )

    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)

    expected = file_signatures([config.baseline_path, config.checkpoint_path, config.skipped_segments_path])
    if manifest.get('dependencies') != expected:
        raise ValueError("Baseline, checkpoint, or skip mask changed. Recalibrate before inspection.")

    if manifest.get('threshold_files') != file_signatures(_threshold_paths(config.calibration_dir)):
        raise ValueError("Threshold files do not match their manifest. Recalibrate before inspection.")


def inspect_board(config: InspectionConfig, board_path: Optional[str] = None, force: bool = False) -> InspectionResults:
    """
    Inspect a processed board and persist results alongside its NumPy file.

    Parameters
    ----------
    config : InspectionConfig
        Shared settings with matching calibration provenance.
    board_path : str, optional
        Board to inspect; defaults to the configured board.
    force : bool
        Recompute even when an unchanged persisted result exists.

    Returns
    -------
    InspectionResults
        Pixel, AE, and intersection grids with source metadata.
    """
    config.resolve()
    path = str(Path(board_path or config.board_path).resolve())
    validate_calibration(config)
    signature = inspection_signature(config, path)
    result_path = Path(path).with_suffix('.inspection.json')
    if not force and result_path.exists():
        try:
            result = InspectionResults.load_json(str(result_path))
            if result.metadata.get('sources') == signature:
                return result
        except (ValueError, KeyError, TypeError):
            pass

    ae_path, pw_path = _threshold_paths(config.calibration_dir)
    result = run_inspection(
        baseline_hexaboard_path=config.baseline_path,
        new_hexaboard_path=path,
        skipped_segments_path=config.skipped_segments_path,
        ae_threshold_path=ae_path,
        pw_threshold_path=pw_path,
        best_model_path=config.checkpoint_path,
        device=config.device,
        model_parameters=config.model
    )
    if signature != inspection_signature(config, path):
        raise ValueError("Inspection inputs changed during processing. Retry inspection.")

    result.metadata['sources'] = signature
    result.save_json(str(result_path))
    return result


def run_inspection(
    baseline_hexaboard_path: str,
    new_hexaboard_path: str,
    skipped_segments_path: str,
    ae_threshold_path: str = './calibrations/ae_threshold.npy',
    pw_threshold_path: str = './calibrations/pw_threshold.npy',
    latent_dim: Optional[int] = None,
    init_filters: Optional[int] = None,
    layers: Optional[List[int]] = None,
    best_model_path: str = './logs/CNNAutoencoder/best/run_01.pt',
    device: Optional[str] = None,
    model_parameters: Optional[Dict] = None
) -> InspectionResults:
    """
    Run both inspection methods using validated boards and threshold grids.

    Parameters
    ----------
    baseline_hexaboard_path : str
        Reference BGR uint8 NumPy board.
    new_hexaboard_path : str
        Processed board to inspect.
    skipped_segments_path : str
        Physical skip mask.
    ae_threshold_path : str
        AE MAE thresholds with shape `[8, 5]`.
    pw_threshold_path : str
        Pixel `1 - SSIM` thresholds with shape `[8, 5]`.
    latent_dim : int, optional
        Checkpoint architecture constraint.
    init_filters : int, optional
        Checkpoint architecture constraint.
    layers : List[int], optional
        Checkpoint architecture constraint.
    best_model_path : str
        CNN weights or trainer checkpoint.
    device : str, optional
        CPU or CUDA device; automatically selected when omitted.
    model_parameters : Dict, optional
        Additional architecture constraints.

    Returns
    -------
    InspectionResults
        Validated status grids. Use `inspect_board` for provenance and caching.
    """
    baseline = open_board(baseline_hexaboard_path)
    board = open_board(new_hexaboard_path)
    skipped = load_skipped_segments(skipped_segments_path)
    thresholds = [np.load(path, allow_pickle=False) for path in (ae_threshold_path, pw_threshold_path)]
    for threshold in thresholds:
        if threshold.shape != baseline.shape[:2] or not np.isfinite(threshold).all():
            raise ValueError("Thresholds must be finite arrays with shape `[8, 5]`.")

    parameters = dict(model_parameters or {})
    for name, value in (('latent_dim', latent_dim), ('init_filters', init_filters), ('layers', layers)):
        if value is not None:
            parameters[name] = value

    resolved_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = load_inspection_model(best_model_path, baseline.shape[2], baseline.shape[3], resolved_device, parameters)
    coordinates = [(r, c) for r in range(baseline.shape[0]) for c in range(baseline.shape[1]) if (r, c) not in skipped]
    pixel_flags = []
    ae_flags = []
    for row, col, pixel, ae in segment_scores(baseline, board, coordinates, model, resolved_device):
        if pixel >= thresholds[1][row, col]:
            pixel_flags.append((row, col))

        if ae >= thresholds[0][row, col]:
            ae_flags.append((row, col))

    return InspectionResults.from_segment_flags(
        baseline.shape[:2],
        pixel_flags,
        ae_flags,
        skipped,
        baseline_hexaboard_path,
        new_hexaboard_path,
        baseline.shape[2:]
    )
