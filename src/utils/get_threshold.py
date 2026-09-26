import json
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

import torch
from torch import nn

from .data import load_skipped_segments
from .data.board_contract import REPO_ROOT, open_board, validate_coordinates
from .inspection_scores import segment_scores


def _as_score_list(value: object) -> List[float]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = [value]

    scores = []
    for candidate in values:
        score = float(candidate)
        if np.isfinite(score):
            scores.append(score)

    return scores


def _select_threshold(good_scores: Sequence[float], bad_scores: Sequence[float]) -> float:
    good = np.asarray(good_scores, dtype=float)
    bad = np.asarray(bad_scores, dtype=float)

    if good.size == 0 or bad.size == 0:
        raise ValueError("Both good_scores and bad_scores must contain at least one finite value.")

    candidates = np.unique(np.concatenate([good, bad]))
    best_key = None
    best_threshold = None

    for threshold in candidates:
        tp = float(np.sum(bad >= threshold))
        fp = float(np.sum(good >= threshold))
        fn = float(np.sum(bad < threshold))
        tpr = tp / bad.size
        fpr = fp / good.size
        denominator = (2.0 * tp) + fp + fn
        f1 = 0.0 if denominator == 0.0 else (2.0 * tp) / denominator
        key = (-f1, fpr, -tpr, -threshold)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)

    if best_threshold is None:
        raise ValueError("Unable to select a threshold from the provided scores.")

    return best_threshold


def process_metrics(
    good_ssims: np.ndarray,
    bad_ssims: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fill_mode: str = 'min'
) -> np.ndarray:
    """
    Compute a per-segment threshold map from good and bad score distributions.

    Parameters
    ----------
    good_ssims : np.ndarray
        Array whose entries are per-segment score lists for good segments. Numeric
        2D arrays are also accepted and are treated as one score per segment.
    bad_ssims : np.ndarray
        Array whose entries are per-segment score lists for bad segments. Numeric
        2D arrays are also accepted and are treated as one score per segment.
    mask : np.ndarray, optional
        Optional boolean mask selecting valid segment positions.
    fill_mode : str
        Method used to fill thresholds outside the mask or where computation
        fails. `'min'` uses the minimum computed threshold, `'max'` uses the
        maximum, and any other value uses the mean.

    Returns
    -------
    np.ndarray
        Array of shape `(H_seg, V_seg)` with a threshold value for each segment.
    """
    good = np.asarray(good_ssims, dtype=object)
    bad = np.asarray(bad_ssims, dtype=object)

    if good.shape != bad.shape:
        raise ValueError(f"good_ssims and bad_ssims must have the same shape, got {good.shape} vs {bad.shape}.")

    H_seg, V_seg = good.shape

    if mask is None:
        inferred_mask = np.zeros((H_seg, V_seg), dtype=bool)
        for h in range(H_seg):
            for v in range(V_seg):
                inferred_mask[h, v] = len(_as_score_list(bad[h, v])) > 0

        mask = inferred_mask

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (H_seg, V_seg):
        raise ValueError(f"mask shape {mask.shape} must match score shape {(H_seg, V_seg)}.")

    thresholds = np.full((H_seg, V_seg), np.nan, dtype=float)

    for h in range(H_seg):
        for v in range(V_seg):
            if not mask[h, v]:
                continue

            good_scores = _as_score_list(good[h, v])
            bad_scores = _as_score_list(bad[h, v])

            if not good_scores or not bad_scores:
                continue

            thresholds[h, v] = _select_threshold(good_scores, bad_scores)

    computed = np.isfinite(thresholds) & mask
    if not computed.any():
        raise ValueError("No valid thresholds could be computed for any masked segment.")

    if fill_mode == 'min':
        fill_value = float(np.nanmin(thresholds[computed]))
    elif fill_mode == 'max':
        fill_value = float(np.nanmax(thresholds[computed]))
    else:
        fill_value = float(np.nanmean(thresholds[computed]))

    thresholds[~computed] = fill_value

    return thresholds


def calibrate_metrics(
    baseline_hexaboard: np.ndarray,
    good_hexaboard_paths: List[str],
    model: nn.Module,
    device: Optional[torch.device] = None,
    json_map_path: str = './calibrations/damaged_segments.json',
    bad_hexaboard_paths: Optional[List[str]] = None,
    skipped_segments: Optional[Set[Tuple[int, int]]] = None,
    normal_quantile: float = 0.99,
    baseline_is_bgr: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Calibrate segment scores while processing one board at a time.

    Parameters
    ----------
    baseline_hexaboard : np.ndarray
        RGB normalized board, or raw uint8 BGR when `baseline_is_bgr=True`.
    good_hexaboard_paths : List[str]
        Normal calibration boards. Each board/coordinate contributes once.
    model : nn.Module
        Trained CNN returning reconstruction logits.
    device : torch.device, optional
        Execution device.
    json_map_path : str
        Damaged-label JSON with repository-relative board IDs.
    bad_hexaboard_paths : List[str], optional
        Explicit anomalous calibration subset. Defaults to labeled bad boards.
    skipped_segments : Set[Tuple[int, int]], optional
        Physical skip mask.
    normal_quantile : float
        Normal-score quantile for positions without defect examples.
    baseline_is_bgr : bool
        Whether the baseline is the raw stored array.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]
        Pixel and AE tuples containing thresholds, bad minima, and good maxima.
    """
    device = device or next(model.parameters()).device
    if baseline_is_bgr:
        baseline = baseline_hexaboard
    else:
        baseline = np.clip(baseline_hexaboard[..., ::-1] * 255, 0, 255).round().astype(np.uint8)

    shape = baseline.shape[:2]
    skipped = (
        load_skipped_segments(shape=shape)
        if skipped_segments is None
        else validate_coordinates(skipped_segments, shape)
    )
    coordinates = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if (r, c) not in skipped]
    with open(json_map_path, 'r', encoding='utf-8') as handle:
        entries = json.load(handle)['files']

    labels = {
        str((REPO_ROOT / path).resolve()): validate_coordinates(
            [(item['row'], item['col']) for item in entry.get('damaged', [])],
            shape
        )
        for path, entry in entries.items()
    }
    good_paths = sorted({str(Path(path).resolve()) for path in good_hexaboard_paths})
    bad_paths = sorted(
        {
            str(Path(path).resolve())
            for path in (
                bad_hexaboard_paths
                if bad_hexaboard_paths is not None
                else [path for path, damaged in labels.items() if damaged]
            )
        }
    )
    if set(good_paths) & set(bad_paths):
        raise ValueError("Normal and anomalous calibration boards must be disjoint.")

    if not good_paths:
        raise ValueError("No normal calibration boards were selected.")

    scores = {(r, c): [[], [], [], []] for r, c in coordinates}
    model.eval()
    for is_bad, paths in ((False, good_paths), (True, bad_paths)):
        for index, path in enumerate(paths):
            damaged = labels.get(path, set())
            if not is_bad and damaged:
                raise ValueError(f"Normal calibration board has damaged labels: `{path}`.")

            if is_bad and not damaged:
                raise ValueError(f"Anomalous calibration board has no damaged labels: `{path}`.")

            selected = [coordinate for coordinate in coordinates if coordinate in damaged] if is_bad else coordinates
            board = open_board(path, baseline.shape)
            for row, col, pixel, ae in segment_scores(baseline, board, selected, model, device):
                offset = 2 if is_bad else 0
                scores[row, col][offset].append(pixel)
                scores[row, col][offset + 1].append(ae)

            print(
                f"Calibration {'bad' if is_bad else 'normal'} board {index + 1}/{len(paths)}: {Path(path).name}",
                flush=True
            )

    metrics = []
    for method in (0, 1):
        thresholds = np.zeros(shape)
        bad_summary = np.zeros(shape)
        good_summary = np.zeros(shape)
        for coordinate, samples in scores.items():
            good = samples[method]
            bad = samples[method + 2]
            if not good:
                raise ValueError(f"No normal scores for segment `{coordinate}`.")

            thresholds[coordinate] = (
                _select_threshold(good, bad) if bad else np.nextafter(float(np.quantile(good, normal_quantile)), np.inf)
            )
            good_summary[coordinate] = max(good)
            bad_summary[coordinate] = min(bad) if bad else 0.0

        metrics.append((thresholds, bad_summary, good_summary))

    return tuple(metrics)
