# Import necessary dependencies
from typing import List, Tuple
from PIL import Image

from preprocessing.process_image import process_image
from pixelwise_inspect.compare_segments import compare_segments
from pixelwise_inspect.calibrate_metrics import calibrate_metrics

# Function to perform pixel-wise comparison
def pw_inference(
    image_paths: List[str],
    num_vertical_segments: int,
    num_horizontal_segments: int
) -> Tuple[List[int], float, List[Image.Image], List[Image.Image]]:
    """
    Performs pixel-wise inference by processing images, segmenting them, and comparing them to identify differences.

    Args:
        image_paths (List[str]): List of paths to the images to be processed.
        num_vertical_segments (int): Number of vertical segments to divide each image into.
        num_horizontal_segments (int): Number of horizontal segments to divide each image into.

    Returns:
        (differences, optimal_threshold, new_segments, baseline_segments) (Tuple[List[int], float, List[Image.Image], List[Image.Image]]):
            - List of indices where differences were found between the first and second image segments.
            - Optimal SSIM threshold for differentiating good and bad segments.
            - List of segmented images from the first image (new segments).
            - List of segmented images from the second image (baseline segments).
    """
    segment_list = []

    # Image processing steps
    for i, image_path in enumerate(image_paths):
        # Read in the image
        image = Image.open(image_path)
        
        # Crop and segment the image based on ArUco markers
        segments, cropped_image = process_image(image, num_vertical_segments, num_horizontal_segments)

        # Save the processed segments (optional)
        # cropped_image.save(os.path.join(DATASET_PATH, 'unperturbed_images', f'hexaboard_{i + 1}.png'))
        
        segment_list.append(segments)

    # Compare the segments of the first two images
    new_segments, baseline_segments = segment_list[0], segment_list[1]
    differences = compare_segments(new_segments, baseline_segments)

    # Calibrate metrics to identify the optimal threshold
    optimal_threshold, bad_values, good_values = calibrate_metrics(new_segments, baseline_segments, differences)

    return differences, optimal_threshold, new_segments, baseline_segments