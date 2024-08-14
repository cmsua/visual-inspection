# Import necessary dependencies
import numpy as np
import cv2

def compare_images(img1_path, img2_path, output_path):
    # Load the two images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if images have the same size and number of channels
    if img1.shape != img2.shape:
        print("Images do not have the same size or number of channels")
        return

    # Compute the difference 
    diff = cv2.absdiff(img1, img2)

    # Enhance the differences
    enhanced_diff = cv2.convertScaleAbs(diff, alpha=3.0, beta=0)  # Increased alpha for more sensitivity

    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(enhanced_diff, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale difference to get the binary mask of differences
    _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)  # Lowered threshold value for more sensitivity

    # Find contours of the differences
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw boxes around differences on the first image
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # threshold
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Save the result
    cv2.imwrite(output_path, img1)
    print(f"Differences highlighted and saved to {output_path}")

# Image paths
img1_path = "/Users/brycewhite/Desktop/TestPictures/Hexa4.png"
img2_path = "/Users/brycewhite/Desktop/TestPictures/Hexa4.1.png"
output_path = "/Users/brycewhite/Desktop/TestPictures/outputsub.png"

compare_images(img1_path, img2_path, output_path)