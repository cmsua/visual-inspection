# Import necessary dependencies
import os
from typing import Tuple, Optional, List
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.image_crop import detect_aruco_markers, bounding_box, crop_to_bounding_box
from preprocessing.adjust_image import adjust_image
from preprocessing.get_segments import get_segments

# Function to return the list of segments and the cropped image
def process_image(
    image: Image.Image,
    num_vertical_segments: int,
    num_horizontal_segments: int
) -> Tuple[Optional[List[Image.Image]], Optional[Image.Image]]:
    """
    Processes an image by detecting ArUco markers, cropping to a bounding box, aligning the image, and segmenting it.

    Args:
        image (Image.Image): Input PIL image to process.
        num_vertical_segments (int): Number of vertical segments to split the image into.
        num_horizontal_segments (int): Number of horizontal segments to split the image into.

    Returns:
        (segments, cropped_image) (Tuple[Optional[List[Image.Image]], Optional[Image.Image]]): A tuple containing 
        a list of segmented image patches and the cropped and aligned image if the processing was successful.
    """
    image = np.array(image)
    corners, ids = detect_aruco_markers(image)
    marker_positions = bounding_box(image, corners, ids)
    cropped_image = crop_to_bounding_box(image, marker_positions)

    # Adjust the images to the correct frame
    cropped_image = Image.fromarray(cropped_image)
    cropped_image = adjust_image(
        image=cropped_image,
        expected_image=cropped_image,
        top_lower_bound=504,
        top_upper_bound=528,
        bottom_lower_bound=532,
        bottom_upper_bound=556,
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

# Run code
if __name__ == "__main__":
    # Path to the datasets folder
    DATASET_PATH = './datasets'

    # Number of vertical and horizontal segments from an image
    NUM_VERTICAL_SEGMENTS = 20
    NUM_HORIZONTAL_SEGMENTS = 12

    # Test the function here
    segments, cropped_image = process_image(
        image=Image.open(os.path.join(DATASET_PATH, 'raw_images', 'hexaboard_01.png')),
        num_vertical_segments=NUM_VERTICAL_SEGMENTS,
        num_horizontal_segments=NUM_HORIZONTAL_SEGMENTS
    )

    # Observe some segments
    if segments:
        # Display the first 8 segments in a 2x4 grid
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(segments):
                ax.imshow(segments[i])
                ax.axis('off')
        plt.suptitle("First 8 Segments of the Image")
        plt.show()

    # Observe the cropped, adjusted image
    if cropped_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_image)
        plt.axis('off')
        plt.title("Cropped and Adjusted Image")
        plt.show()