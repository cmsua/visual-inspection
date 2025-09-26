from typing import Dict, List, Tuple

import numpy as np

import torch
from torch import Tensor


def accuracy_metric_bce(outputs: Tensor, target: Tensor) -> float:
    """
    Computes accuracy for binary classification when using BCEWithLogitsLoss.
    Applies sigmoid on raw logits, then thresholds at 0.5.
    """
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().view(-1)
    target = target.view(-1)
    
    return torch.sum(preds == target).item() / target.size(0)


def accuracy_metric_ce(pred: Tensor, target: Tensor) -> float:
    r"""
    Accuracy for multi-class classification (using CrossEntropyLoss).
    """
    pred_class = pred.argmax(dim=1)

    # Check if target is one-hot encoded
    if target.dim() > 1 and target.shape[1] > 1:
        target_class = target.argmax(dim=1)  # convert one-hot to class indices
    else:
        target_class = target  # already class indices

    return torch.sum(pred_class == target_class).item() / target.size(0)


def agg_confusion_matrix(
    bad_hexaboard_paths: List[str],
    pred_good_list: List[List[Tuple[int, int]]],
    pred_bad_list: List[List[Tuple[int, int]]],
    true_bad_map: Dict[str, List[Tuple[int, int]]],
    total_per_board: int
) -> np.ndarray:
    # Returns a 2x2 numpy array [[TN, FP], [FN, TP]] aggregated over all boards
    TN = FP = FN = TP = 0

    # Good boards: all segments are actually negative; any flagged is FP
    for flagged in pred_good_list:
        fset = set(flagged)
        FP += len(fset)
        TN += total_per_board - len(fset)

    # Bad boards: compare flagged vs true_bad for each board (assume filename order preserved)
    for idx, flagged in enumerate(pred_bad_list):
        fset = set(flagged)

        # Find corresponding true bad set by filename order using bad_hexaboard_paths
        fname = bad_hexaboard_paths[idx] if idx < len(bad_hexaboard_paths) else None
        true_set = set(true_bad_map.get(fname, [])) if fname is not None else set()

        TP_b = len(fset & true_set)
        FP_b = len(fset - true_set)
        FN_b = len(true_set - fset)
        TN_b = total_per_board - (TP_b + FP_b + FN_b)

        TP += TP_b
        FP += FP_b
        FN += FN_b
        TN += TN_b

    return np.array([[TN, FP], [FN, TP]], dtype=int)