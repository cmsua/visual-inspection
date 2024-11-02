# Import necessary dependencies
import os

import numpy as np
import cv2

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