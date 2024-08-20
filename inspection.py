# Import necessary dependencies
import os
import sys
import argparse
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from autoencoder.image_lineup import adjust_image
from autoencoder.data_loading import RotationAndSegmentationTransform, HexaboardDataset, SimpleCNNAutoEncoder
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

# class ScanResult():
#     def __init__(self, base: np.ndarray, annotated: np.ndarray = None) -> None:
#         self.base = base
#         self.annotated = annotated

# def run_inspection(image: Image.Image) -> ScanResult:
#     # Adjust the image to the correct frame
#     image = adjust_image(
#         image,
#         top_lower_bound=378,
#         top_upper_bound=402,
#         bottom_lower_bound=401,
#         bottom_upper_bound=425,
#         bound_range=24,
#         num_channels=3,
#         view=True
#     )

#     return image

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
    image_dir = os.path.join(DATASET_PATH, 'raw_data')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    # remove_transparency(image_dir)  # only for first-time usage

    # List of segments where segmentlist[i] is a list of all segments from one image
    segmentList = []

    # Image processing steps
    for image_path in image_paths:
        # Read in the images
        image = Image.open(image_path)
        
        # Crop the images based on ArUco markers
        segments, cropped_image = process_image(image, NUM_VERTICAL_SEGMENTS, NUM_HORIZONTAL_SEGMENTS)
        segmentList.append(segments)

    print(len(segmentList), type(segmentList[0]))

    
        
    # # Define the transformations
    # height, width, _ = cropped_image.shape
    # transform = transforms.Compose([
    #     RotationAndSegmentationTransform(
    #         height=height,
    #         width=width,
    #         vertical_segments=NUM_VERTICAL_SEGMENTS,
    #         horizontal_segments=NUM_HORIZONTAL_SEGMENTS
    #     ),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    # ])

    # # Read in and process the images
    # dataset = HexaboardDataset(
    #     image_dir=image_dir,
    #     transform=transform
    # )
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # # Get the segments' height and width
    # segment_height = dataset[0][0][0].shape[0]
    # segment_width = dataset[0][0][0].shape[1]

    # # Load the autoencoder model
    # criterion = nn.BCEWithLogitsLoss()
    # model = SimpleCNNAutoEncoder(
    #     height=segment_height,
    #     width=segment_width,
    #     latent_dim=128,
    #     kernel_sizes=[64, 128]
    # )
    # model.to(device)
    # model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # # Evaluate the segments
    # total_loss = 0

    # with torch.no_grad():
    #     for inp in dataloader:
    #         for input in inp:
    #             input = input.to(device)
    #             output = model(input)
    #             loss = criterion(output, input)
    #             total_loss += loss.item()

    # test_loss = total_loss / len(dataloader)
    # print(f'Test Loss: {test_loss:.4f}')