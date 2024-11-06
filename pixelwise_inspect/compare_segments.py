# Import necessary dependencies
from typing import List
from PIL import Image

import numpy as np
import cv2

# Function for pixel-by-pixel comparison
def compare_segments(segments1: List[Image.Image], segments2: List[Image.Image]) -> List[int]:
    """
    Compares two lists of image segments and identifies the indices where differences are found.

    Args:
        segments1 (List[np.ndarray]): The first list of image segments.
        segments2 (List[np.ndarray]): The second list of image segments.

    Returns:
        differences (List[int]): A list of indices where differences are found between the two sets of image segments.
    """
    assert len(segments1) == len(segments2), "Segment lists are not of the same length"

    differences = []

    for i, (seg1, seg2) in enumerate(zip(segments1, segments2)):
        img1 = np.array(seg1)
        img2 = np.array(seg2)

        # Check if the segment shapes are different
        if img1.shape != img2.shape:
            differences.append(i)
            continue

        # Calculate absolute difference and enhance it
        diff = cv2.absdiff(img1, img2)
        enhanced_diff = cv2.convertScaleAbs(diff, alpha=3.0, beta=0)

        # Convert to grayscale and apply threshold to create a binary mask
        gray_diff = cv2.cvtColor(enhanced_diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

        # Find contours to identify regions with significant differences
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            differences.append(i)

    return differences