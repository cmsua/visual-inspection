# Import necessary dependencies
import os
import sys
import shutil
from PIL import Image

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix

from autoencoder.image_lineup import adjust_image

# Directory path used in local
project_dir = './'
autoencoder_dir = os.path.join(project_dir, 'autoencoder')
sys.path.append(autoencoder_dir)

# Paths
DATASET_PATH = os.path.join(project_dir, 'datasets')
RESULT_PATH = os.path.join(project_dir, 'results')

# Function to load the image
def load_image(image_path: str):
    return cv2.imread(image_path)

# Function to detect the ArUco markers
def detect_aruco_markers(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    return corners, ids

# Function to draw the bounding box
def bounding_box(image, corners, ids):
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        marker_positions = [np.mean(corner[0], axis=0) for corner in corners]

        if len(marker_positions) == 4:
            marker_positions = np.array(marker_positions)
            marker_positions = marker_positions[np.argsort(np.arctan2(
                marker_positions[:, 1] - np.mean(marker_positions[:, 1]),
                marker_positions[:, 0] - np.mean(marker_positions[:, 0])
            ))]

            for i in range(4):
                start_point = tuple(marker_positions[i].astype(int))
                end_point = tuple(marker_positions[(i + 1) % 4].astype(int))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)

            return marker_positions
        
    return None

# Function to crop the image based on the bounding box
def crop_to_bounding_box(image, marker_positions):
    if marker_positions is not None:
        pts = np.array(marker_positions, dtype=np.float32)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = image[y:y + h, x:x + w].copy()

        return cropped
    
    return None

# Function to get segments from the hexaboard
def get_segments(image, height: int, width: int, vertical_segments: int, horizontal_segments: int, rotation: int | None = None):
    a = width / 2
    
    # Transform and crop image
    transform = None
    if rotation is None: 
        transform = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.CenterCrop((height, int(a)))
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.RandomRotation(degrees=(rotation, rotation)),
            transforms.Resize((height, width)),
            transforms.CenterCrop((height, int(a)))
        ])

    cropped_image = transform(image)

    # Calculate the dimensions of each segment
    segment_height = int(height / vertical_segments)
    segment_width = int(a / horizontal_segments)

    segments = []

    # Split the cropped image into subsegments
    for i in range(vertical_segments):
        for j in range(horizontal_segments):
            left = j * segment_width
            upper = i * segment_height
            right = left + segment_width
            lower = upper + segment_height

            segment = cropped_image.crop((left, upper, right, lower))
            segments.append(segment)

    return segments

# Function for pixel-by-pixel comparison
def compare_segments(segments1, segments2):
    assert len(segments1) == len(segments2), "Segment lists are not of the same length"

    differences = []

    for i, (seg1, seg2) in enumerate(zip(segments1, segments2)):
        img1 = np.array(seg1)
        img2 = np.array(seg2)

        if img1.shape != img2.shape:
            differences.append(i)
            continue

        diff = cv2.absdiff(img1, img2)
        enhanced_diff = cv2.convertScaleAbs(diff, alpha=3.0, beta=0)
        gray_diff = cv2.cvtColor(enhanced_diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            differences.append(i)

    return differences

# Function to remove the transparency channel
def remove_transparency(
    image_dir: str,
):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        img = Image.open(image_path)
        img = np.array(img)
        img = img[:, :, 0:3]
        img = Image.fromarray(img)
        img.save(image_path)

# Function to return the list of segments and the cropped image
def process_image(image: Image.Image, num_vertical_segments: int, num_horizontal_segments: int):
    image = np.array(image)
    corners, ids = detect_aruco_markers(image)
    marker_positions = bounding_box(image, corners, ids)
    cropped_image = crop_to_bounding_box(image, marker_positions)

    # Adjust the images to the correct frame
    cropped_image = Image.fromarray(cropped_image)
    cropped_image = adjust_image(
        image=cropped_image,
        expected_image=cropped_image,
        top_lower_bound=378,
        top_upper_bound=402,
        bottom_lower_bound=401,
        bottom_upper_bound=425,
        bound_range=24,
        num_channels=3,
        view=False
    )

    # Get the segments from each image
    if cropped_image is not None:
        width, height = cropped_image.size
        segments_1 = get_segments(cropped_image, height, width, num_vertical_segments, num_horizontal_segments)
        segments_2 = get_segments(cropped_image, height, width, num_vertical_segments, num_horizontal_segments, 60)
        segments_3 = get_segments(cropped_image, height, width, num_vertical_segments, num_horizontal_segments, 300)
        segments = segments_1 + segments_2 + segments_3

        return segments, cropped_image

    return None, None

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

if __name__ == "__main__":
    # Paths for input images and output comparison result
    image_path = os.path.join(DATASET_PATH, 'raw_images', 'hexaboard_01')
    reference_image_path = os.path.join(DATASET_PATH, 'raw_images', 'hexaboard_02')

    # Read the images
    image = Image.open(image_path)
    reference_image = Image.open(reference_image_path)

    # Ensure the save directory exists
    save_dir = os.path.join(RESULT_PATH, 'segments')
    os.makedirs(save_dir, exist_ok=True)

    NUM_VERTICAL_SEGMENTS = 20
    NUM_HORIZONTAL_SEGMENTS = 12

    # Load and process the images
    segments1, cropped_image1 = process_image(image, NUM_VERTICAL_SEGMENTS, NUM_HORIZONTAL_SEGMENTS)
    segments2, cropped_image2 = process_image(reference_image, NUM_VERTICAL_SEGMENTS, NUM_HORIZONTAL_SEGMENTS)

    # Compare the segments and print differences
    differences = compare_segments(segments1, segments2)
    print(f"Found differences in segments: {differences}")

    bad_ssims = []
    good_ssims = []

    for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):
        measure_value = evaluate_inspection(segments1, segments2)

        if i in differences:
            bad_ssims.append(measure_value)
        else:
            good_ssims.append(measure_value)

    process_inspection(good_ssims, bad_ssims)