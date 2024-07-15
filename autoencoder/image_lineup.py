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
    max_rotation: float = 0.25
) -> Image.Image:
    # Random shifts
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    # Random rotation
    rotation = random.uniform(-max_rotation, max_rotation)

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

    plt.imshow(output.squeeze(0).moveaxis(0, 2).detach().cpu().numpy())
    plt.show()

    best_match_location = (output == max_value).nonzero(as_tuple=True)
    batch_idx, channel_idx, y_coord, x_coord = best_match_location
    y_coord += lower_y_bound
    x_coord += lower_x_bound
    print(x_coord, y_coord)

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
    aligned_image = torchvision.transforms.functional.affine(
        image,
        angle=angle,
        translate=(translation_x, translation_y),
        scale=1,
        shear=0
    )

    return aligned_image

# Function to find the area with T-shape
def view_T_shape(
    image: Image.Image,
    x_lb: int, x_ub: int,
    y_lb: int, y_ub: int,
    title: str = None
):
    # Convert the image to a numpy array for indexing
    image_array = np.array(image)

    # Extract the region
    region = image_array[y_lb:y_ub, x_lb:x_ub]

    # Display the extracted region
    plt.imshow(region)
    if title:
        plt.title(title)
    else:
        plt.title("Region around T-shape")
    plt.show()

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
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0]
]

bottom_T_shape = [
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1]
]

# Run Code
if __name__ == "__main__":
    # Load the unperturbed image
    image = Image.open(os.path.join(DATASET_PATH, 'unperturbed_data', 'HexaBoardExample.png'))

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

    # Adjust based on expected T-shape location
    top_x_lb, top_x_ub = 380, 400
    top_y_lb, top_y_ub = 0, 20
    bottom_x_lb, bottom_x_ub = 403, 423
    bottom_y_lb, bottom_y_ub = image.height - 20, image.height

    # # View the T-shapes of the based image
    # view_T_shape(
    #     image=image,
    #     x_lb=top_x_lb, x_ub=top_x_ub,
    #     y_lb=top_y_lb, y_ub=top_y_ub
    # )
    # view_T_shape(
    #     image=image,
    #     x_lb=bottom_x_lb, x_ub=bottom_x_ub,
    #     y_lb=bottom_y_lb, y_ub=bottom_y_ub
    # )

    # Load the transformed image
    transformed_image = Image.open(os.path.join(DATASET_PATH, 'transformed_data', 'transformed_image_1.png'))
    print("Shape of the Transformed Image:", transformed_image.size)

    # # View the T-shapes of the transformed image
    # view_T_shape(
    #     image=transformed_image,
    #     x_lb=top_x_lb, x_ub=top_x_ub,
    #     y_lb=top_y_lb, y_ub=top_y_ub
    # )
    # view_T_shape(
    #     image=transformed_image,
    #     x_lb=bottom_x_lb, x_ub=bottom_x_ub,
    #     y_lb=bottom_y_lb, y_ub=bottom_y_ub
    # )

    # Generate the kernels
    num_channels = 3
    top_T_kernel = generate_T_kernel(top_T_shape, num_channels)
    bottom_T_kernel = generate_T_kernel(bottom_T_shape, num_channels)

    # Detect top and bottom T-shapes for the based image
    expected_top_x, expected_top_y = detect_T_shape(image, top_x_lb, top_x_ub, top_y_lb, top_y_ub, top_T_kernel)
    expected_bottom_x, expected_bottom_y = detect_T_shape(image, bottom_x_lb, bottom_x_ub, bottom_y_lb, bottom_y_ub, bottom_T_kernel)

    # Hard-code the coordinates to the center of the T-shapes
    expected_top_x = expected_top_x + 2
    expected_bottom_x = expected_bottom_x + 2
    expected_bottom_y = expected_bottom_y - 6

    # Detect top and bottom T-shapes for the transformed image
    top_x, top_y = detect_T_shape(transformed_image, top_x_lb, top_x_ub, top_y_lb, top_y_ub, top_T_kernel)
    bottom_x, bottom_y = detect_T_shape(transformed_image, bottom_x_lb, bottom_x_ub, bottom_y_lb, bottom_y_ub, bottom_T_kernel)

    # Hard-code the coordinates to the center of the T-shapes
    top_x = top_x + 2
    bottom_x = bottom_x + 2
    bottom_y = bottom_y - 6

    # Align the transformed image
    aligned_image = align_image(
        transformed_image,
        top_x, top_y, bottom_x, bottom_y,
        expected_top_x, expected_top_y, expected_bottom_x, expected_bottom_y
    )

    # View the T-shapes of the based image and the realigned image for comparisons
    view_T_shape(
        image=image,
        x_lb=top_x_lb, x_ub=top_x_ub,
        y_lb=top_y_lb, y_ub=top_y_ub,
        title='Top T of Based Image'
    )
    view_T_shape(
        image=aligned_image,
        x_lb=top_x_lb, x_ub=top_x_ub,
        y_lb=top_y_lb, y_ub=top_y_ub,
        title='Top T of Aligned Image'
    )
    view_T_shape(
        image=image,
        x_lb=bottom_x_lb, x_ub=bottom_x_ub,
        y_lb=bottom_y_lb, y_ub=bottom_y_ub,
        title='Bottom T of Based Image'
    )
    view_T_shape(
        image=aligned_image,
        x_lb=bottom_x_lb, x_ub=bottom_x_ub,
        y_lb=bottom_y_lb, y_ub=bottom_y_ub,
        title='Bottom T of Aligned Image'
    )