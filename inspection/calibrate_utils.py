# Import necessary dependencies
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_curve, roc_curve

# Function to determine the optimal_threshold
def process_inspection(good_ssims: np.ndarray, bad_ssims: np.ndarray) -> float:
    """
    Processes the inspection by plotting SSIM histograms, calculating precision-recall and ROC curves, 
    and finding the optimal SSIM threshold.

    Args:
        good_ssims (np.ndarray): Array of SSIM values for good image segments.
        bad_ssims (np.ndarray): Array of SSIM values for bad image segments.

    Returns:
        optimal_threshold (float): The optimal SSIM threshold that minimizes the false negative rate (1 - TPR).
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