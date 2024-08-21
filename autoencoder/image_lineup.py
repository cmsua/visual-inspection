# Import necessary dependencies
import os
import random
from PIL import Image
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision

# Path to the datasets folder
DATASET_PATH = './datasets'

# Function to randomly rotate and shift the image
def random_tranform(
    image: Image.Image,
    max_shift: int = 5,
    max_rotation: float = 0.05
) -> Image.Image:
    # Random shifts
    shift_x = random.randint(1, max_shift) * random.choice([-1, 1])
    shift_y = random.randint(1, max_shift) * random.choice([-1, 1])

    # Random rotation
    rotation = random.uniform(.025, max_rotation) * random.choice([-1, 1])

    # Apply the shift
    shifted_image = torchvision.transforms.functional.affine(
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
    # Convert PIL image to PyTorch tensor
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
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
    kernel = torch.tensor(t_shape, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(1, num_channels, 1, 1)

    return kernel

# Define the T-shape kernels
top_T_shape = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
]

bottom_T_shape = [
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# Function to take in an image and return an adjusted image
def adjust_image(
    image: Image.Image,
    expected_image: Image.Image,
    top_lower_bound: int,
    top_upper_bound: int,
    bottom_lower_bound: int,
    bottom_upper_bound: int,
    bound_range: int,
    num_channels: int = 3,
    view: bool = False
) -> Image.Image:
    # Adjust based on expected T-shape location
    top_x_lb, top_x_ub = top_lower_bound, top_upper_bound
    top_y_lb, top_y_ub = 0, bound_range
    bottom_x_lb, bottom_x_ub = bottom_lower_bound, bottom_upper_bound
    bottom_y_lb, bottom_y_ub = image.height - bound_range, image.height

    # Generate the kernels
    top_T_kernel = generate_T_kernel(top_T_shape, num_channels)
    bottom_T_kernel = generate_T_kernel(bottom_T_shape, num_channels)

    # Detect top and bottom T-shapes for the based image
    expected_top_x, expected_top_y = detect_T_shape(expected_image, top_x_lb, top_x_ub, top_y_lb, top_y_ub, top_T_kernel)
    expected_bottom_x, expected_bottom_y = detect_T_shape(expected_image, bottom_x_lb, bottom_x_ub, bottom_y_lb, bottom_y_ub, bottom_T_kernel)

    # Hard-code the coordinates to the center of the T-shapes
    expected_bottom_y = expected_bottom_y - bottom_T_kernel.shape[-2] + 1

    # Detect top and bottom T-shapes for the transformed image
    top_x, top_y = detect_T_shape(image, top_x_lb, top_x_ub, top_y_lb, top_y_ub, top_T_kernel)
    bottom_x, bottom_y = detect_T_shape(image, bottom_x_lb, bottom_x_ub, bottom_y_lb, bottom_y_ub, bottom_T_kernel)

    # Hard-code the coordinates to the center of the T-shapes
    bottom_y = bottom_y - bottom_T_kernel.shape[-2] + 1

    # Align the transformed image
    adjusted_image = align_image(
        image, top_x, top_y, bottom_x, bottom_y,
        expected_top_x, expected_top_y, expected_bottom_x, expected_bottom_y
    )

    # Visualize the adjustments (optional)
    if view:
        img_arr = np.array(image)
        adj_img_arr = np.array(adjusted_image)
        exp_img_arr = np.array(expected_image)

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))

        axs[0, 0].imshow(img_arr[top_y_lb:top_y_ub, top_x_lb:top_x_ub])
        axs[0, 0].axis('off')

        axs[0, 1].imshow(adj_img_arr[top_y_lb:top_y_ub, top_x_lb:top_x_ub])
        axs[0, 1].axis('off')

        axs[0, 2].imshow(exp_img_arr[top_y_lb:top_y_ub, top_x_lb:top_x_ub])
        axs[0, 2].axis('off')

        axs[1, 0].imshow(img_arr[bottom_y_lb:bottom_y_ub, bottom_x_lb:bottom_x_ub])
        axs[1, 0].axis('off')

        axs[1, 1].imshow(adj_img_arr[bottom_y_lb:bottom_y_ub, bottom_x_lb:bottom_x_ub])
        axs[1, 1].axis('off')

        axs[1, 2].imshow(exp_img_arr[bottom_y_lb:bottom_y_ub, bottom_x_lb:bottom_x_ub])
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    return adjusted_image

# Run code
if __name__ == "__main__":
    # Load the unperturbed image
    image = Image.open(os.path.join(DATASET_PATH, 'unperturbed_data', 'good_hexaboard.png'))

    # # Display the entire image
    # plt.imshow(image)
    # plt.title("Full HexaBoard Image")
    # plt.show()

    # # Generate and display random transformations
    # transformed_images = [random_tranform(image) for _ in range(4)]

    # # Save the images
    # for i, img in enumerate(transformed_images):
    #     img.save(os.path.join(DATASET_PATH, 'transformed_data', f"transformed_image_{i+1}.png"))

    # # Display the images
    # fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # for ax, img in zip(axes, transformed_images):
    #     ax.imshow(img)
    #     ax.axis('off')

    # plt.show()

    # Load the transformed image
    transformed_image = Image.open(os.path.join(DATASET_PATH, 'transformed_data', 'transformed_image_1.png'))
    
    print("Shape of the Transformed Image:", transformed_image.size)

    # Adjust the transformed image
    adjusted_image = adjust_image(
        image=transformed_image,
        expected_image=image,
        top_lower_bound=378,
        top_upper_bound=402,
        bottom_lower_bound=401,
        bottom_upper_bound=425,
        bound_range=24,
        num_channels=3,
        view=True
    )