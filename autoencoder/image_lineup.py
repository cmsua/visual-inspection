# Import necessary dependencies
import os
import random
from PIL import Image
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Path to the datasets folder
DATASET_PATH = './datasets'

# Function to randomly rotate and shift the image
def random_tranform(
    image: Image.Image,
    max_shift: int = 10,
    max_rotation: float = 1.0
) -> Image.Image:
    width, height = image.size

    # Random shifts
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    # Random rotation
    rotation = random.uniform(-max_rotation, max_rotation)

    # Apply the shift
    shifted_image = Image.new('RGB', (width + abs(shift_x), height + abs(shift_y)))
    shifted_image.paste(image, (max(shift_x, 0), max(shift_y, 0)))
    shifted_image = shifted_image.crop(
        (
            abs(shift_x),
            abs(shift_y),
            abs(shift_x) + width,
            abs(shift_y) + height
        )
    )

    # Apply the rotation
    rotated_image = shifted_image.rotate(rotation, expand=True)

    return rotated_image

# Function to detect the T-shape pattern using convolution
def detect_T_shape(
    image: Image.Image,
    lower_x_bound: int,
    upper_x_bound: int,
    lower_y_bound: int,
    upper_y_bound: int,
    kernel: torch.Tensor
) -> Tuple[int, int]:
    # Convert PIL image to PyTorch tensor
    image_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0).permute(0, 3, 1, 2)
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
    # Calculate the translation needed to align the top T-shape
    translation_x = expected_top_x - top_x
    translation_y = expected_top_y - top_y
    
    # Apply the translation
    translated_image = Image.new('RGB', (image.width + abs(translation_x), image.height + abs(translation_y)))
    translated_image.paste(image, (max(translation_x, 0), max(translation_y, 0)))
    translated_image = translated_image.crop(
        (
            abs(translation_x),
            abs(translation_y),
            abs(translation_x) + image.width,
            abs(translation_y) + image.height
        )
    )
    
    # Calculate the rotation needed to align the bottom T-shape
    angle = np.arctan2(expected_bottom_y - expected_top_y, expected_bottom_x - expected_top_x) - np.arctan2(bottom_y - top_y, bottom_x - top_x)
    angle = np.degrees(angle)
    
    # Apply the rotation
    aligned_image = translated_image.rotate(angle, center=(expected_top_x, expected_top_y), expand=True)
    
    return aligned_image

# Function to generate the convolution kernel
def generate_T_kernel(
    t_shape: List[List[int]],
    num_channels: int
) -> torch.Tensor:
    kernel = torch.tensor(t_shape, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(1, num_channels, 1, 1)

    return kernel

# Define the T-shape kernels
top_t_shape = [
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0]
]

bottom_t_shape = [
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1]
]

# Load the image
image = Image.open(os.path.join(DATASET_PATH, 'HexaBoardExample.png'))

# Display the entire image
plt.imshow(image)
plt.title("Full HexaBoard Image")
plt.show()

# Convert the image to a numpy array for indexing
image_array = np.array(image)

# Define the region around the top-left T-shape
# Adjust these bounds based on visual inspection of the image
lower_x_bound, upper_x_bound = 380, 400
lower_y_bound, upper_y_bound = 0, 20

# Extract the region
top_left_region = image_array[lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound]

# Display the extracted region
plt.imshow(top_left_region)
plt.title("Top Left Region around T-shape")
plt.show()

# # Generate and display random transformations
# transformed_images = [random_tranform(image) for _ in range(4)]

# # Save the images
# for i, img in enumerate(transformed_images):
#     img.save(os.path.join(DATASET_PATH, f"transformed_image_{i+1}.png"))

# # Display the images
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# for ax, img in zip(axes, transformed_images):
#     ax.imshow(img)
#     ax.axis('off')

# plt.show()