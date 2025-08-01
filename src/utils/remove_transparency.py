import os
from PIL import Image

import numpy as np


def remove_transparency(image_dir: str) -> None:
    """
    Removes the transparency channel from all PNG images in the specified directory.

    Parameters
    ----------
    image_dir : str
        Directory containing PNG images.
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        img = Image.open(image_path)
        img = np.array(img)
        img = img[:, :, 0:3]
        img = Image.fromarray(img)
        img.save(image_path)