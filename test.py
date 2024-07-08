import cv2
import numpy as np

# Load the image
image_path = "/Users/brycewhite/Desktop/TestPictures/Hexa4.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour which should be the hexagonal board
largest_contour = max(contours, key=cv2.contourArea)

# Create a blank mask with the same dimensions as the image
mask = np.zeros_like(image)

# Draw the largest contour on the mask
cv2.drawContours(mask, [largest_contour], -1, (0, 255, 0), 2)

# Overlay the mask on the original image
result = cv2.addWeighted(image, 1, mask, 1, 0)

# Calculate the perimeter of the contour
perimeter = cv2.arcLength(largest_contour, True)

# Add perimeter text to the image
cv2.putText(result, f'Perimeter: {perimeter:.2f} pixels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the result
cv2.imshow('Hexagonal Board', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
