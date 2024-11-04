# Import necessary dependencies
import os
from PIL import Image

import numpy as np

# Function to remove the transparency layer (used when manually perturbed the images)
def remove_transparency(image_dir: str) -> None:
    """
    Removes the transparency channel from all PNG images in the specified directory.

    Args:
        image_dir (str): Directory containing PNG images to process.
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        img = Image.open(image_path)
        img = np.array(img)
        img = img[:, :, 0:3]
        img = Image.fromarray(img)
        img.save(image_path)