import cv2
import numpy as np

# Load the image
image_path = "/Users/brycewhite/Desktop/TestPictures/HexaArUcoTest2.png"
image = cv2.imread(image_path)

# Define the dictionary of ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Detect the markers in the image
corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

if ids is not None:
    # Draw detected markers on the image
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Extract the center coordinates of the markers
    marker_positions = []
    for corner in corners:
        center = np.mean(corner[0], axis=0)
        marker_positions.append(center)

    # Ensure we have exactly 4 markers
    if len(marker_positions) == 4:
        # Sort marker positions to maintain a consistent order
        marker_positions = np.array(marker_positions)
        marker_positions = marker_positions[np.argsort(np.arctan2(marker_positions[:,1] - np.mean(marker_positions[:,1]), marker_positions[:,0] - np.mean(marker_positions[:,0])))]
        
        # Draw lines between the marker positions to form a bounding box
        for i in range(4):
            start_point = tuple(marker_positions[i].astype(int))
            end_point = tuple(marker_positions[(i+1) % 4].astype(int))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # Calculate the perimeter of the bounding box
        perimeter = 0
        for i in range(4):
            start_point = marker_positions[i]
            end_point = marker_positions[(i+1) % 4]
            perimeter += np.linalg.norm(end_point - start_point)

        # Add perimeter text to the image
        cv2.putText(image, f'Perimeter: {perimeter:.2f} pixels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the result
cv2.imshow('Hexagonal Board with Bounding Box', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
