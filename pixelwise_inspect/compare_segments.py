# Import necessary dependencies
import os

import numpy as np
import cv2

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