# Import necessary dependencies
import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from adjust_utils import detect_T_shape, align_image, generate_T_kernel  

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
