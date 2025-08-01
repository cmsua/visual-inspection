from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_curve, roc_curve


def process_inspection(good_ssims: np.ndarray, bad_ssims: np.ndarray) -> float:
    """
    Processes the inspection by plotting SSIM histograms, calculating precision-recall and ROC curves, 
    and finding the optimal SSIM threshold.

    Parameters
    ----------
    good_ssims : np.ndarray
        Array of SSIM values for good image segments.
    bad_ssims : np.ndarray
        Array of SSIM values for bad image segments.
    
    Returns
    -------
    float
        The optimal SSIM threshold that minimizes the false negative rate (1 - TPR).
    """
    
    # # Plot histograms of SSIM values for good and bad segments
    # _, bins, _ = plt.hist(good_ssims, density=True, histtype='step', bins=30, label='Good Event SSIM')
    # plt.hist(bad_ssims, density=True, bins=bins, histtype='step', label='Bad Event SSIM', color='red')
    # plt.xlabel('SSIM')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()

    # Concatenate SSIM values and create labels
    ssim_values = np.concatenate([good_ssims, bad_ssims])
    labels = np.concatenate([np.ones_like(good_ssims), np.zeros_like(bad_ssims)])

    # Calculate precision-recall and ROC curves
    # precision, recall, pr_thresholds = precision_recall_curve(labels, ssim_values)
    fpr, tpr, roc_thresholds = roc_curve(labels, ssim_values)

    # Find the optimal threshold that minimizes the false negative rate (1 - TPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    # print(f"Optimal Threshold using SSIM: {optimal_threshold:.4f}")

    return optimal_threshold


def calibrate_metrics(
    baseline_hexaboard: np.ndarray,
    perturbed_hexaboard: np.ndarray,
    second_baseline_hexaboard: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calibrates metrics by calculating SSIM values for pairs of hexaboard segments, distinguishing
    between 'bad' segments (perturbed vs. baseline) and 'good' segments (baseline vs. second baseline).
    Determines the optimal SSIM threshold for segment classification.

    Parameters
    ----------
    baseline_hexaboard : np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing the baseline hexaboard.
    perturbed_hexaboard : np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing a perturbed hexaboard.
    second_baseline_hexaboard : np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing a second baseline hexaboard for comparison.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        A tuple containing:
        - optimal_threshold (float): The optimal SSIM threshold for segment classification.
        - bad_ssims (np.ndarray): SSIM values for 'bad' segments (perturbed vs. baseline).
        - good_ssims (np.ndarray): SSIM values for 'good' segments (baseline vs. second baseline).
    """
    
    if baseline_hexaboard.ndim != 5 or perturbed_hexaboard.ndim != 5 or second_baseline_hexaboard.ndim != 5:
        raise ValueError("All hexaboard arrays must be 5D (H_seg, V_seg, height, width, num_channels)")
    
    H_seg, V_seg, _, _, _ = baseline_hexaboard.shape
    
    # Normalize to [0, 1] range
    baseline_hexaboard = baseline_hexaboard.astype(np.float32) / 255.0
    perturbed_hexaboard = perturbed_hexaboard.astype(np.float32) / 255.0
    second_baseline_hexaboard = second_baseline_hexaboard.astype(np.float32) / 255.0
    
    bad_ssims = []
    good_ssims = []

    # Calculate SSIM for each segment pair (baseline vs perturbed) - these are "bad" comparisons
    for h in range(H_seg):
        for v in range(V_seg):
            baseline_segment = baseline_hexaboard[h, v]
            perturbed_segment = perturbed_hexaboard[h, v]
            ssim_val = ssim(baseline_segment, perturbed_segment, data_range=1.0, channel_axis=2)
            bad_ssims.append(ssim_val)

    # Calculate SSIM for each segment pair (baseline vs second baseline) - these are "good" comparisons
    for h in range(H_seg):
        for v in range(V_seg):
            baseline_segment = baseline_hexaboard[h, v]
            second_baseline_segment = second_baseline_hexaboard[h, v]
            ssim_val = ssim(baseline_segment, second_baseline_segment, data_range=1.0, channel_axis=2)
            good_ssims.append(ssim_val)

    bad_ssims = np.array(bad_ssims)
    good_ssims = np.array(good_ssims)

    # Process the inspection to find the optimal SSIM threshold
    optimal_threshold = process_inspection(good_ssims, bad_ssims)

    return optimal_threshold, bad_ssims, good_ssims