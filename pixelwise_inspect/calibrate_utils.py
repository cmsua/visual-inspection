import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_curve, roc_curve

# Function to evaluate the inspection using SSIM
def evaluate_inspection(segment1, segment2):
    # Load images
    img1 = np.array(segment1)
    img2 = np.array(segment2)

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    measure_value, _ = ssim(img1_gray, img2_gray, full=True)

    return measure_value

# Function to flag bad segments and save them
def process_inspection(good_ssims, bad_ssims):
    # Plot histograms of SSIM values
    _, bins, _ = plt.hist(good_ssims, density=1, histtype='step', bins=30, label='Good Event SSIM')
    plt.hist(bad_ssims, density=1, bins=bins, histtype='step', label='Bad Event SSIM',color='red')
    plt.xlabel('SSIM')

    #plt.xlim()
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Concatenate SSIMS value & create labels
    ssim = np.concatenate([good_ssims, bad_ssims])
    labels1 = np.concatenate([np.ones_like(good_ssims), np.zeros_like(bad_ssims)])

    # Precision Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(labels1, ssim)
    fpr, tpr, roc_thresholds = roc_curve(labels1, ssim)

    # Find the threshold that minimizes FNR (1 - TPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    print(f"Optimal Threshold using SSIM: {optimal_threshold}")

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
    