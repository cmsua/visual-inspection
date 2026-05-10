from typing import Tuple

import pytest

import torch
from torch import nn, optim

from src.models import CNNAutoencoder
from src.utils import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config() -> Tuple:
    input_channels = 3
    height = 1060
    width = 1882
    latent_dim = 16
    init_filters = 32
    layers = [2, 2, 2]

    return input_channels, height, width, latent_dim, init_filters, layers


def test_forward_output_shape(config: Tuple) -> None:
    in_ch, height, width, latent_dim, init_filters, layers = config
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)
    x = torch.randn(1, in_ch, height, width, device=device)
    output = model(x)

    # Catch any shape mismatch
    assert output.shape == x.shape


def test_forward_no_nan(config: Tuple) -> None:
    in_ch, height, width, latent_dim, init_filters, layers = config
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)
    x = torch.randn(1, in_ch, height, width, device=device)
    output = model(x)

    # Ensure no NaNs
    assert not torch.isnan(output).any()


def test_training_step(config: Tuple) -> None:
    in_ch, height, width, latent_dim, init_filters, layers = config
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(1, in_ch, height, width, device=device)
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, x)
    loss.backward()
    optimizer.step()

    # Loss should be a positive scalar
    assert isinstance(loss.item(), float) and loss.item() >= 0


def test_bottleneck_geometry(config: Tuple) -> None:
    _, height, width, latent_dim, init_filters, layers = config
    model = CNNAutoencoder(
        height=height,
        width=width,
        latent_dim=latent_dim,
        init_filters=init_filters,
        layers=layers
    ).to(device)

    assert model.bottleneck_height == 9
    assert model.bottleneck_width == 16
    assert model.bottleneck_area < 160
    assert model.bottleneck_height > 1 and model.bottleneck_width > 1
    assert abs((model.bottleneck_height / model.bottleneck_width) - (height / width)) < 0.1
