import numpy as np

from src.utils.get_threshold import process_metrics


def test_process_metrics_optimizes_f1_score() -> None:
    good_scores = np.empty((1, 1), dtype=object)
    bad_scores = np.empty((1, 1), dtype=object)
    good_scores[0, 0] = [0.10, 0.40]
    bad_scores[0, 0] = [0.50, 0.60]
    mask = np.array([[True]], dtype=bool)

    thresholds = process_metrics(good_scores, bad_scores, mask=mask, fill_mode='min')

    assert thresholds.shape == (1, 1)
    assert thresholds[0, 0] == 0.50


def test_process_metrics_prefers_lower_fpr_when_f1_ties() -> None:
    good_scores = np.empty((1, 1), dtype=object)
    bad_scores = np.empty((1, 1), dtype=object)
    good_scores[0, 0] = [0.20, 0.40]
    bad_scores[0, 0] = [0.50, 0.70]
    mask = np.array([[True]], dtype=bool)

    thresholds = process_metrics(good_scores, bad_scores, mask=mask, fill_mode='min')

    assert thresholds[0, 0] == 0.50


def test_process_metrics_prefers_higher_tpr_when_f1_and_fpr_tie() -> None:
    good_scores = np.empty((1, 1), dtype=object)
    bad_scores = np.empty((1, 1), dtype=object)
    good_scores[0, 0] = [0.45, 0.90]
    bad_scores[0, 0] = [0.50, 0.70, 0.90]
    mask = np.array([[True]], dtype=bool)

    thresholds = process_metrics(good_scores, bad_scores, mask=mask, fill_mode='min')

    assert thresholds[0, 0] == 0.50


def test_process_metrics_fill_mode_uses_minimum_computed_threshold() -> None:
    good_scores = np.empty((1, 2), dtype=object)
    bad_scores = np.empty((1, 2), dtype=object)
    good_scores[0, 0] = [0.10, 0.40]
    bad_scores[0, 0] = [0.50, 0.60]
    good_scores[0, 1] = []
    bad_scores[0, 1] = []
    mask = np.array([[True, False]], dtype=bool)

    thresholds = process_metrics(good_scores, bad_scores, mask=mask, fill_mode='min')

    assert thresholds[0, 0] == 0.50
    assert thresholds[0, 1] == 0.50
