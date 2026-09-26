from typing import Dict, Optional

import torch
from torch import nn

from ..models import CNNAutoencoder


def load_inspection_model(
    path: str,
    height: int,
    width: int,
    device: torch.device,
    parameters: Optional[Dict] = None
) -> nn.Module:
    """
    Load a CNN checkpoint after verifying its architecture.

    Parameters
    ----------
    path : str
        State dictionary or trainer checkpoint.
    height : int
        Segment height.
    width : int
        Segment width.
    device : torch.device
        Execution device.
    parameters : Dict, optional
        Explicit architecture constraints to check against the weights.

    Returns
    -------
    nn.Module
        CNN autoencoder in evaluation mode.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    try:
        inferred = {
            'init_filters': int(state['conv1.weight'].shape[0]),
            'latent_dim': int(state['encoder_layers.2.0.0.weight'].shape[0]),
            'layers': [
                len({key.split('.')[2] for key in state if key.startswith(f'encoder_layers.{stage}.')})
                for stage in range(3)
            ]
        }
    except (KeyError, AttributeError) as exc:
        raise ValueError("Checkpoint is not compatible with `CNNAutoencoder`.") from exc

    for key, value in (parameters or {}).items():
        expected = {'height': height, 'width': width, **inferred}.get(key)
        if value != expected:
            raise ValueError(f"Model setting `{key}={value}` does not match checkpoint/input value `{expected}`.")

    model = CNNAutoencoder(height=height, width=width, **inferred).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
