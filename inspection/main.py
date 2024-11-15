# Import necessary dependencies
import os
import json
import argparse
from PIL import Image

import matplotlib.pyplot as plt
import torch

from preprocessing.process_image import process_image
from pixelwise_inspect.pw_inference import pw_inference
from autoencoder.ae_inference import ae_inference
# from inspection.opt_sort import opt_sort

# Directory path used in local
project_dir = './'
autoencoder_dir = os.path.join(project_dir, 'autoencoder')

# Paths
RESULT_PATH = os.path.join(project_dir, 'results')
CHECKPOINT_PATH = os.path.join(autoencoder_dir, 'small_ae.pt')

# Specify the device to use the autoencoder model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# INSPECTION PROCESS
# RUN THIS COMMAND FROM visual-inspection
# python -m inspection.main -n "datasets/raw_images/hexaboard_01.png" -b "datasets/raw_images/hexaboard_02.png" -t "inspection/calibration.json" -vs 20 -hs 12
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Visual Inspection',
        description='Inspects Hexaboards for defects',
        epilog='University of Alabama'
    )

    # Set up arguments
    parser.add_argument('-n', '--new_image_path', type=str, help='path to new image')
    parser.add_argument('-b', '--baseline_image_path', type=str, help='path to baseline image')
    parser.add_argument('-t', '--threshold_path', type=str, help='optimal threshold for SSIM')
    parser.add_argument('-vs', '--vertical_segments', type=int, help='number of vertical image segments')
    parser.add_argument('-hs', '--horizontal_segments', type=int, help='number of horizontal image segments')
    
    # Parse and retrieve the arguments
    args = parser.parse_args()
    new_image = Image.open(args.new_image_path)
    baseline_image = Image.open(args.baseline_image_path)
    with open(args.threshold_path, 'r') as fin:
        threshold, bad_ssims, good_ssims = json.load(fin)

    # Get all segments from the new and baseline images
    new_segments, _ = process_image(new_image, args.vertical_segments, args.horizontal_segments)
    baseline_segments, _= process_image(baseline_image, args.vertical_segments, args.horizontal_segments)

    # Perform inferences
    pw_indices = pw_inference(new_segments, baseline_segments, threshold)
    ae_indices = ae_inference(new_segments, threshold, device, CHECKPOINT_PATH)

    # Lists of flagged segments
    double_flagged = sorted(list(set(pw_indices) & set(ae_indices)))
    ml_flagged = set(ae_indices) - set(double_flagged) 
    pixel_flagged = set(pw_indices) - set(double_flagged)
    all_flagged = sorted(list(set(ae_indices).union(pw_indices)))
    print(double_flagged)