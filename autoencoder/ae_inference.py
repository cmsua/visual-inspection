# Import necessary dependencies
from typing import List
from PIL import Image

import numpy as np
from torchvision.transforms import ToTensor
import torch
from skimage.metrics import structural_similarity as ssim

from autoencoder.model import SimpleCNNAutoEncoder

# Function to flag bad segments using autoencoder
def ae_inference(
    newSegments: List[Image.Image],
    threshold: float,
    device: torch.device,
    checkpoint_path: str
) -> List[int]:
    """
    Performs inference using an autoencoder model to evaluate segments of an image and flags segments with
    SSIM values below a specified threshold, indicating potential anomalies.

    Args:
        newSegments (List[Image.Image]): List of segmented images to evaluate.
        threshold (float): SSIM threshold below which segments are flagged as differing significantly.
        device (torch.device): The device (CPU or GPU) to load the model and perform inference.
        checkpoint_path (str): Path to the saved model checkpoint.

    Returns:
        flagged_indices (List[int]): List of indices of segments with SSIM values below the threshold, flagged as anomalous.
    """
    segment_width, segment_height = newSegments[0].size
    
    model = SimpleCNNAutoEncoder(
        height=segment_height,
        width=segment_width,
        latent_dim=128,
        kernel_sizes=[64, 128]
    )
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Evaluate the segments
    flagged_indices = []

    for i, seg1 in enumerate(newSegments):
        # output should be compared to seg1, not seg2?
        seg1_tensor = ToTensor()(seg1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(seg1_tensor).cpu().squeeze().permute(1, 2, 0).numpy()

            # SSIM metric
            ssim_val = ssim(output, np.array(seg1), channel_axis=2, data_range=output.max() - output.min())

            if ssim_val < threshold:
                flagged_indices.append(i)

    return flagged_indices