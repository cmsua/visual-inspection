import os

from PIL import Image
import numpy as np

from image_crop import detect_aruco_markers, bounding_box, crop_to_bounding_box
from adjust_image import adjust_image
from get_segments import get_segments

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