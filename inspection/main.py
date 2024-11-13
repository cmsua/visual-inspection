# Import necessary dependencies
import os
import sys

import torch

from pixelwise_inspect.pw_inference import pw_inference
from autoencoder.ae_inference import ae_inference
# from inspection.opt_sort import opt_sort

# Directory path used in local
project_dir = './'
autoencoder_dir = os.path.join(project_dir, 'autoencoder')
sys.path.append(autoencoder_dir)

# Paths
DATASET_PATH = os.path.join(project_dir, 'datasets')
RESULT_PATH = os.path.join(project_dir, 'results')
CHECKPOINT_PATH = os.path.join(autoencoder_dir, 'small_ae.pt')

# Specify the device to use the autoencoder model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     prog='Visual Inspection',
    #     description='Inspects Hexaboards for defects',
    #     epilog='University of Alabama'
    # )

    # # Set up arguments
    # parser.add_argument('folder')
    # parser.add_argument('-v', '--verbose', action='store_true')

    # # Parse
    # args = parser.parse_args()
    
    # Adjust the number of segments
    # THIS SHOULD WORK WITH THE GUI
    NUM_VERTICAL_SEGMENTS = 20
    NUM_HORIZONTAL_SEGMENTS = 12

    # Get the directory to all images
    image_dir = os.path.join(DATASET_PATH, 'raw_images')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    # remove_transparency(image_dir)  # only for first-time usage

    flaggedP, pw_optimal_threshold, pw_values, newSegments, baselineSegments,  = pw_inference(image_paths, NUM_VERTICAL_SEGMENTS, NUM_HORIZONTAL_SEGMENTS)

    # pw_opt_good, pw_opt_flagged = opt_sort(pw_optimal_threshold, pw_values)
    # flaggedP = pw_opt_flagged

    # bad_ssims = []
    # good_ssims = []

    # for i, (segment1, segment2) in enumerate(zip(newSegments, baselineSegments)):
    #     measure_value = evaluate_inspection(newSegments, baselineSegments)

    #     if i in flaggedP:
    #         bad_ssims.append(measure_value)
    #     else:
    #         good_ssims.append(measure_value)

    # process_inspection(good_ssims, bad_ssims)

    flaggedML, ae_optimal_threshold, ae_values = ae_inference(device, CHECKPOINT_PATH, newSegments, baselineSegments)

    # ae_opt_good, ae_opt_flagged = opt_sort(ae_optimal_threshold, ae_values)
    # flagged_ML = ae_opt_flagged

    double_flagged = sorted(list(set(flaggedP) & set(flaggedML)))
    ml_flagged = set(flaggedML) - set(double_flagged) 
    pixel_flagged = set(flaggedP) - set(double_flagged)
    all_flagged = sorted(list(set(flaggedML).union(flaggedP)))
    print(double_flagged)