# Import necessary dependencies
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_curve, roc_curve

# Function to evaluate the inspection using SSIM
def evaluate_inspection(segment1: np.ndarray, segment2: np.ndarray) -> float:
    """
    Evaluates the similarity between two image segments using the Structural Similarity Index (SSIM).

    Args:
        segment1 (np.ndarray): The first image segment.
        segment2 (np.ndarray): The second image segment.

    Returns:
        measure_value (float): The SSIM value indicating the similarity between the two segments (1 indicates perfect similarity).
    """
    # Convert to grayscale
    img1_gray = cv2.cvtColor(segment1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(segment2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    measure_value, _ = ssim(img1_gray, img2_gray, full=True)

    return measure_value

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
    precision, recall, pr_thresholds = precision_recall_curve(labels, ssim_values)
    fpr, tpr, roc_thresholds = roc_curve(labels, ssim_values)

    # Find the optimal threshold that minimizes the false negative rate (1 - TPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    print(f"Optimal Threshold using SSIM: {optimal_threshold:.4f}")

    return optimal_threshold

    # double_flag = []

    # for i, ssim in enumerate(bad_ssims):
    #     if ssim <= optimal_threshold:
    #         element = differences[i]
    #         double_flag.append(element)

    # print(double_flag)
    
    # for i in differences:
    #     filename = f'segment1_{i}.png'
    #     file_to_copy = os.path.join(RESULT_PATH, 'segments', filename)
    #     destination = os.path.join(RESULT_PATH, 'flagged_segments')
    #     shutil.copy(file_to_copy, destination)

    # for i in double_flag:
    #     filename = f'segment1_{i}.png'
    #     file_to_copy = os.path.join(RESULT_PATH, 'segments', filename)
    #     destination = os.path.join(RESULT_PATH, 'db_flagged_segments')
    #     shutil.copy(file_to_copy, destination)
    