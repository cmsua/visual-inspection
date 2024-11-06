# Import necessary dependencies
import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.adjust_utils import random_tranform, detect_T_shape, align_image, generate_T_kernel

# Path to the datasets folder
DATASET_PATH = './datasets'

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
    """
    Adjusts an image to align it based on detected T-shape patterns by matching them to the expected image.

    Args:
        image (Image.Image): The input image to adjust.
        expected_image (Image.Image): The reference image with the expected T-shape positions.
        top_lower_bound (int): Lower bound of the x-coordinate range for detecting the top T-shape.
        top_upper_bound (int): Upper bound of the x-coordinate range for detecting the top T-shape.
        bottom_lower_bound (int): Lower bound of the x-coordinate range for detecting the bottom T-shape.
        bottom_upper_bound (int): Upper bound of the x-coordinate range for detecting the bottom T-shape.
        bound_range (int): Height range for detecting the top and bottom T-shapes in the image.
        num_channels (int, optional): Number of channels in the image. Defaults to 3.
        view (bool, optional): Whether to visualize the alignment process. Defaults to False.

    Returns:
        adjusted_image (Image.Image): The adjusted image aligned with the reference image.
    """
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
    # Load the perturbed image
    image = Image.open(os.path.join(DATASET_PATH, 'perturbed_images', 'hexaboard_1.png'))

    # # Generate and display random transformations
    # transformed_images = [random_tranform(image) for _ in range(4)]

    # # Save the images
    # for i, img in enumerate(transformed_images):
    #     img.save(os.path.join(DATASET_PATH, 'transformed_images', f"transformed_image_{i+1}.png"))

    # # Display the images
    # fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # for ax, img in zip(axes, transformed_images):
    #     ax.imshow(img)
    #     ax.axis('off')

    # plt.show()

    # Load the transformed image
    transformed_image = Image.open(os.path.join(DATASET_PATH, 'transformed_images', 'transformed_image_1.png'))

    # Display the entire image
    plt.imshow(transformed_image)
    plt.title("Full Hexaboard Image")
    plt.show()
    
    print("Shape of the Transformed Image:", transformed_image.size)

    # Adjust the transformed image
    adjusted_image = adjust_image(
        image=transformed_image,
        expected_image=image,
        top_lower_bound=504,
        top_upper_bound=528,
        bottom_lower_bound=532,
        bottom_upper_bound=556,
        bound_range=24,
        num_channels=3,
        view=True
    )