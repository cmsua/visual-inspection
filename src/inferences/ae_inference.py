from typing import List, Tuple, Union

import numpy as np
from skimage.metrics import structural_similarity as ssim

import torch

from ..models import ResNetAutoencoder


@torch.no_grad()
def autoencoder_inference(
    hexaboard: np.ndarray,
    threshold: float,
    latent_dim: int = 128,
    init_filters: int = 64,
    layers: List[int] = [2, 2, 2],
    best_model_path: str = './logs/ResNetAutoencoder/best/run_01.pt',
    device: Union[torch.device, str] = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> List[Tuple[int, int]]:
    """
    Compares the autoencoder's reconstruction of hexaboard segments against the original segments and
    flags every (H_seg_idx, V_seg_idx) segment whose reconstruction SSIM drops below *threshold*.

    Parameters
    ----------
    hexaboard : np.ndarray
        5D array with shape (H_seg, V_seg, height, width, num_channels) representing a single hexaboard.
    threshold : float
        The SSIM threshold below which segments are flagged.
    device : Union[torch.device, str]
        The device to run the inference on.
    best_model_path : str
        The path to the model checkpoint.
    latent_dim : int, optional
        Dimension of the latent (bottleneck) vector.
    init_filters : int, optional
        Number of filters in the first convolutional layer (ResNet stem).
    layers : List[int], optional
        Number of BasicBlock modules in each ResNet stage.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing the (H_seg_idx, V_seg_idx) indices of the flagged segments.
    """

    if hexaboard.ndim != 5:
        raise ValueError(f"Expected 5D array (H_seg, V_seg, height, width, num_channels), got shape {hexaboard.shape}")
    
    H_seg, V_seg, height, width, num_channels = hexaboard.shape
    device = torch.device(device) if isinstance(device, str) else device
    
    # Load model
    model = ResNetAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Create dataset
    hexaboard = torch.tensor(hexaboard.astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3)
    flagged_segments = []

    for h in range(H_seg):
        for v in range(V_seg):
            segment = hexaboard[h, v].unsqueeze(0)  # (1, num_channels, height, width)
            segment = segment.to(device, non_blocking=device.type == 'cuda')
            recons = torch.sigmoid(model(segment))
            pred = recons[0].permute(1, 2, 0).cpu().numpy()  # (height, width, num_channels)
            true = segment[0].permute(1, 2, 0).cpu().numpy()
            ssim_val = ssim(pred, true, data_range=1.0, channel_axis=2)

            if ssim_val < threshold:
                flagged_segments.append((h, v))

    return flagged_segments