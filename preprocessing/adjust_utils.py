# Import necessary dependencies
import random
from PIL import Image
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision

# Function to randomly rotate and shift the image
def random_tranform(
    image: Image.Image,
    max_shift: int = 5,
    max_rotation: float = 0.05
) -> Image.Image:
    """
    Applies a random shift and rotation to an image.

    Args:
        image (Image.Image): The input PIL image.
        max_shift (int, optional): Maximum number of pixels to shift the image in x and y directions. Defaults to 5.
        max_rotation (float, optional): Maximum rotation angle in radians. Defaults to 0.05.

    Returns:
        shifted_image (Image.Image): The transformed image with random shift and rotation.
    """
    # Random shifts
    shift_x = random.randint(1, max_shift) * random.choice([-1, 1])
    shift_y = random.randint(1, max_shift) * random.choice([-1, 1])

    # Random rotation
    rotation = random.uniform(.025, max_rotation) * random.choice([-1, 1])

    # Apply the shift
    shifted_image = TF.affine(
        image,
        angle=rotation,
        translate=(shift_x, shift_y),
        scale=1,
        shear=0
    )

    return shifted_image

# Function to detect the T-shape pattern using convolution
def detect_T_shape(
    image: Image.Image,
    lower_x_bound: int,
    upper_x_bound: int,
    lower_y_bound: int,
    upper_y_bound: int,
    kernel: torch.Tensor
) -> Tuple[int, int]:
    """
    Detects the T-shape pattern in a specified region of the image using convolution.

    Args:
        image (Image.Image): The input PIL image.
        lower_x_bound (int): Lower bound for x-coordinate of the region.
        upper_x_bound (int): Upper bound for x-coordinate of the region.
        lower_y_bound (int): Lower bound for y-coordinate of the region.
        upper_y_bound (int): Upper bound for y-coordinate of the region.
        kernel (torch.Tensor): Convolution kernel to detect the T-shape.

    Returns:
        (x_coord, y_coord) (Tuple[int, int]): Coordinates (x, y) of the best match location for the T-shape pattern.
    """
    # Convert PIL image to PyTorch tensor
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    top_corner_image = image_tensor[:, :, lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound]
    output = F.conv2d(top_corner_image, kernel)
    max_value = output.max()
    best_match_location = (output == max_value).nonzero(as_tuple=True)
    batch_idx, channel_idx, y_coord, x_coord = best_match_location

    y_coord += lower_y_bound
    x_coord += lower_x_bound

    return x_coord.item(), y_coord.item()

# Function to align the image based on detected T-shape patterns
def align_image(
    image: Image.Image, top_x: int, top_y: int,
    bottom_x: int, bottom_y: int, expected_top_x: int,
    expected_top_y: int, expected_bottom_x: int , expected_bottom_y: int
) -> Image.Image:
    """
    Aligns the image based on detected T-shape patterns by calculating necessary translation and rotation.

    Args:
        image (Image.Image): The input PIL image to align.
        top_x (int): Detected x-coordinate of the top T-shape.
        top_y (int): Detected y-coordinate of the top T-shape.
        bottom_x (int): Detected x-coordinate of the bottom T-shape.
        bottom_y (int): Detected y-coordinate of the bottom T-shape.
        expected_top_x (int): Expected x-coordinate of the top T-shape.
        expected_top_y (int): Expected y-coordinate of the top T-shape.
        expected_bottom_x (int): Expected x-coordinate of the bottom T-shape.
        expected_bottom_y (int): Expected y-coordinate of the bottom T-shape.

    Returns:
        aligned_image (Image.Image): The aligned image.
    """
    # Calculate the translation needed to align the top T-shape
    translation_x = expected_top_x - top_x
    translation_y = expected_top_y - top_y

    # Calculate the rotation needed to align the bottom T-shape
    angle = np.arctan2(expected_bottom_y - expected_top_y, expected_bottom_x - expected_top_x) - np.arctan2(bottom_y - top_y, bottom_x - top_x)
    angle = np.degrees(angle)

    # Calculate the center of rotation
    center_of_rotation_x = top_x + translation_x
    center_of_rotation_y = top_y + translation_y

    # Calculate the translation back to original top T-shape position after rotation
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    back_translation_x = center_of_rotation_x * (1 - cos_angle) + center_of_rotation_y * sin_angle
    back_translation_y = center_of_rotation_y * (1 - cos_angle) - center_of_rotation_x * sin_angle

    # Combine the initial translation, rotation, and back translation into a single affine transform
    combined_translation_x = translation_x + back_translation_x
    combined_translation_y = translation_y + back_translation_y

    aligned_image = torchvision.transforms.functional.affine(
        image,
        angle=angle,
        translate=(combined_translation_x, combined_translation_y),
        scale=1,
        shear=0,
        center=(center_of_rotation_x, center_of_rotation_y)
    )

    return aligned_image

# Function to generate the convolution kernel
def generate_T_kernel(
    t_shape: List[List[int]],
    num_channels: int
) -> torch.Tensor:
    """
    Generates a convolution kernel to detect a T-shape pattern in an image.

    Args:
        t_shape (List[List[int]]): 2D list representing the T-shape pattern.
        num_channels (int): Number of channels for the kernel (matches image channels).

    Returns:
        kernel (torch.Tensor): The generated T-shape convolution kernel.
    """
    kernel = torch.tensor(t_shape, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(1, num_channels, 1, 1)

    return kernel