# Import necessary dependencies
import os

from calibrate_utils import evaluate_inspection, process_inspection

# Directory path used in local
project_dir = './'
# autoencoder_dir = os.path.join(project_dir, 'autoencoder')
# sys.path.append(autoencoder_dir)

# Paths
DATASET_PATH = os.path.join(project_dir, 'datasets')
RESULT_PATH = os.path.join(project_dir, 'results')

def calibrate_metrics(segments1, segments2, differences):

    # Ensure the save directory exists
    save_dir = os.path.join(RESULT_PATH, 'segments')
    os.makedirs(save_dir, exist_ok=True)

    bad_ssims = []
    good_ssims = []
    bad_segments = []
    good_segments = []

    for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):
        measure_value = evaluate_inspection(segments1, segments2)

        if i in differences:
            bad_ssims.append(measure_value)
            bad_segments.append(segments1, segments2)
        else:
            good_ssims.append(measure_value)
            good_segments.append(segments1, segments2)

    optimal_threshold = process_inspection(good_ssims, bad_ssims)
    
    bad_values = zip(bad_ssims, bad_segments)
    good_values = zip(good_ssims, good_segments)

    return optimal_threshold, bad_values, good_values
