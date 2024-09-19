# Import necessary dependencies
import os
import sys
from PIL import Image

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import torch
from torchvision.transforms import ToTensor

from autoencoder import SimpleCNNAutoEncoder
from utils import *

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

    # List of segments where segmentlist[i] is a list of all segments from one image
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
    segment_width, segment_height = newSegments[0].size

    # List of different segments based on indices
    flaggedP = compare_segments(newSegments, baselineSegments)

    # bad_ssims = []
    # good_ssims = []

    # for i, (segment1, segment2) in enumerate(zip(newSegments, baselineSegments)):
    #     measure_value = evaluate_inspection(newSegments, baselineSegments)

    #     if i in flaggedP:
    #         bad_ssims.append(measure_value)
    #     else:
    #         good_ssims.append(measure_value)

    # process_inspection(good_ssims, bad_ssims)

    # Load the autoencoder model
    model = SimpleCNNAutoEncoder(
        height=segment_height,
        width=segment_width,
        latent_dim=128,
        kernel_sizes=[64, 128]
    )
    model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Evaluate the segments
    flaggedML = []
    threshold = 0.4

    for i, (seg1, seg2) in enumerate(zip(newSegments, baselineSegments)):
        seg1 = ToTensor()(seg1).unsqueeze(0)
        seg2 = np.array(seg2)

        with torch.no_grad():
            output = model(seg1).cpu().squeeze().permute(1, 2, 0).numpy()
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            seg2 = cv2.cvtColor(seg2, cv2.COLOR_BGR2GRAY)

            # SSIM metric
            ssim_val, _ = ssim(output, seg2, full=True, data_range=output.max() - output.min())

            if ssim_val < threshold:
                flaggedML.append(i)

    double_flagged = sorted(list(set(flaggedP) & set(flaggedML)))
    ml_flagged = set(flaggedML) - set(double_flagged) 
    pixel_flagged = set(flaggedP) - set(double_flagged)
    all_flagged = sorted(list(set(flaggedML).union(flaggedP)))
    print(double_flagged)