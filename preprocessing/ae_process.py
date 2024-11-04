# Import necessary libraries
import os
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from get_segments import get_segments

# Define the custom transformation for rotations and segmentations
class RotationAndSegmentationTransform:
    """
    Custom transformation class that applies random rotations and flips to the image, then segments it into multiple 
    parts based on specified grid dimensions.
    
    Args:
        height (int): The height of each segment.
        width (int): The width of each segment.
        vertical_segments (int): Number of vertical segments to split the image into.
        horizontal_segments (int): Number of horizontal segments to split the image into.
        tf (Optional[transforms.Compose], optional): A custom transform composition for additional augmentations. 
                                                     Defaults to None, which applies default transformations.
    
    Returns:
        torch.Tensor: A tensor containing the stacked segments of the transformed image.
    """
    def __init__(
        self,
        height: int,
        width: int,
        vertical_segments: int,
        horizontal_segments: int,
        tf: Optional[transforms.Compose] = None
    ):
        self.height: int = height
        self.width: int = width
        self.vertical_segments: int = vertical_segments
        self.horizontal_segments: int = horizontal_segments

        # If no custom transformation is provided, use default transformations
        if tf is None:
            self.tf: transforms.Compose = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.tf = tf

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply random transformations (rotation, flip) to the image and segment it into multiple parts.
        
        Args:
            image (Image.Image): The input PIL image.
        
        Returns:
            torch.Tensor: A tensor of stacked segments of the transformed image.
        """
        a = self.width / 2  # Semi-width for hexagonal calculations
        height = int(np.sqrt(3) * a)  # Compute height based on hexagon geometry

        segments: list[torch.Tensor] = []

        # Get the segments with different rotations (0, 60, and 300 degrees)
        segments.extend(get_segments(image, height, self.width, self.vertical_segments, self.horizontal_segments))
        segments.extend(get_segments(image, height, self.width, self.vertical_segments, self.horizontal_segments, 60))
        segments.extend(get_segments(image, height, self.width, self.vertical_segments, self.horizontal_segments, 300))

        # Apply transformations to each segment
        segments = [self.tf(segment) for segment in segments]

        # Stack all transformed segments into a single tensor
        return torch.stack(segments)

# Define the custom dataset
class HexaboardDataset(Dataset):
    """
    Custom Dataset class for loading and transforming hexaboard images.

    Args:
        image_dir (str): Directory path where the hexaboard images are stored.
        transform (Optional[RotationAndSegmentationTransform], optional): A transformation function to apply to each image.
                                                                         Defaults to None, which applies no transformations.

    Returns:
        torch.Tensor: The transformed image or its segments if transformations are applied.
    """
    def __init__(self, image_dir: str, transform: Optional[RotationAndSegmentationTransform] = None) -> None:
        self.image_dir = image_dir
        self.transform = transform

        # Get all image paths in the directory ending with .png
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Loads an image, applies the transformation if specified, and returns it.

        Args:
            idx (int): Index of the image to be fetched.

        Returns:
            torch.Tensor: The transformed image (or segments of it) in tensor format.
        """
        # Ensure the index is within range (supports cycling)
        actual_idx = idx % len(self.image_paths)
        img_path = self.image_paths[actual_idx]
        
        # Open the image
        try:
            image = Image.open(img_path)
        except IOError:
            raise RuntimeError(f"Failed to open image at {img_path}")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image