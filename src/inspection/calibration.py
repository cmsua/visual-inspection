import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import torch

from .artifacts import file_signatures, save_json
from .model import load_inspection_model
from ..configs.inspection_config import InspectionConfig
from ..utils.data import load_skipped_segments
from ..utils.data.board_contract import open_board
from ..utils.get_threshold import calibrate_metrics
from ..utils.multigpu import set_seed


def calibration_sources(config: InspectionConfig) -> Dict[str, List[str]]:
    with open(config.labels_path, 'r', encoding='utf-8') as handle:
        entries = json.load(handle)['files']

    damaged_ids = {Path(key).stem for key, value in entries.items() if value.get('damaged')}
    good = sorted(
        {
            str(path.resolve())
            for directory in config.good_dirs
            for path in Path(directory).glob('*.npy')
            if path.resolve() != Path(config.baseline_path) and path.stem not in damaged_ids
        }
    )
    bad_by_id = {
        path.stem: str(path.resolve())
        for path in sorted(Path(config.bad_dir).glob('*.npy'))
        if path.stem in damaged_ids
    }
    bad = sorted(bad_by_id.values())
    if not good or not bad:
        raise ValueError("Calibration requires normal boards and annotated anomalous boards.")

    rng = np.random.default_rng(config.seed)
    selected = rng.permutation(len(bad))[: max(1, math.ceil(len(bad) * config.bad_fraction))]
    return {'good': good, 'bad': sorted(bad[int(index)] for index in selected)}


def calibrate(config: InspectionConfig) -> Dict:
    """
    Generate threshold arrays and provenance for server inspection.

    Parameters
    ----------
    config : InspectionConfig
        Shared inspection settings, including calibration data selection.

    Returns
    -------
    Dict
        Saved calibration manifest with model and data provenance.
    """
    config.resolve()
    set_seed(config.seed)
    baseline = open_board(config.baseline_path)
    skipped = load_skipped_segments(config.skipped_segments_path)
    sources = calibration_sources(config)
    device = torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = load_inspection_model(config.checkpoint_path, baseline.shape[2], baseline.shape[3], device, config.model)
    pixel, ae = calibrate_metrics(
        baseline,
        sources['good'],
        model,
        device,
        config.labels_path,
        bad_hexaboard_paths=sources['bad'],
        skipped_segments=skipped,
        normal_quantile=config.normal_quantile,
        baseline_is_bgr=True
    )
    destination = Path(config.calibration_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for prefix, values, names in (
        ('pw', pixel, ('threshold', 'bad_ssims', 'good_ssims')),
        ('ae', ae, ('threshold', 'bad_maes', 'good_maes'))
    ):
        for name, array in zip(names, values):
            path = destination / f'{prefix}_{name}.npy'
            with path.with_suffix('.npy.tmp').open('wb') as handle:
                np.save(handle, array)

            path.with_suffix('.npy.tmp').replace(path)
            outputs[f'{prefix}_{name}'] = str(path)

    manifest = {
        'version': 1,
        'score': {'pixel': '1-SSIM', 'autoencoder': 'MAE', 'comparison': '>='},
        'board_shape': list(baseline.shape),
        'config': asdict(config),
        'sources': sources,
        'dependencies': file_signatures([config.baseline_path, config.checkpoint_path, config.skipped_segments_path]),
        'threshold_files': file_signatures([outputs['pw_threshold'], outputs['ae_threshold']]),
        'data_files': file_signatures(sources['good'] + sources['bad'] + [config.labels_path])
    }
    save_json(str(destination / 'manifest.json'), manifest)
    print(f"Saved calibration to {destination}", flush=True)
    return manifest
