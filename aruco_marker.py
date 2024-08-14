# Import necessary dependencies
import numpy as np
import cv2

# Function to generate ArUco marker
def generate_aruco_marker(marker_id, size=133, dictionary=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    return marker_image

if __name__ == "__main__":
    # Generate and save 4 ArUco markers with IDs 0 to 3
    for marker_id in range(4):
        cv2.imwrite(f'aruco_marker_{marker_id}.png', generate_aruco_marker(marker_id))