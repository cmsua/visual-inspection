import os
from PIL import Image

import numpy as np

from preprocessing.process_image import process_image
from compare_segments import compare_segments
from calibrate_metrics import calibrate_metrics

def pw_inference(image_paths, NUM_VERTICAL_SEGMENTS, NUM_HORIZONTAL_SEGMENTS):
    segmentList = []

    # Image processing steps
    for i, image_path in enumerate(image_paths):
            # Read in the images
            image = Image.open(image_path)
        
            # Crop the images based on ArUco markers
            segments, cropped_image = process_image(image, NUM_VERTICAL_SEGMENTS, NUM_HORIZONTAL_SEGMENTS)
            # cropped_image.save(os.path.join(DATASET_PATH, 'unperturbed_images', f'hexaboard_{i + 1}.png'))
            segmentList.append(segments)

    newSegments, baselineSegments = segmentList[0], segmentList[1]
    differences = compare_segments(newSegments, baselineSegments)

    optimal_threshold, bad_values, good_values = calibrate_metrics(newSegments, baselineSegments, differences)
    #values = np.concatenate([bad_values, good_values])
    
    return differences, optimal_threshold, newSegments, baselineSegments
