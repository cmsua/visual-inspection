from typing import List, Tuple

import torch
from torch import nn, Tensor
from torchvision.models.resnet import BasicBlock


def _conv_out(h: int, k: int, s: int, p: int, d: int = 1) -> int:
    """Standard conv output formula (same as PyTorch docs)."""
    return (h + 2 * p - d * (k - 1) - 1) // s + 1


def _deconv_out(h: int, k: int, s: int, p: int, op: int) -> int:
    """ConvTranspose2d output formula."""
    return (h - 1) * s - 2 * p + k + op


class ResNetAutoencoder(nn.Module):
    r"""
    A ResNet-inspired CNN autoencoder using ConvTranspose2d.

    Parameters
    ----------
    height : int
        Height of the input images.
    width : int
        Width of the input images.
    latent_dim : int
        Dimension of the latent (bottleneck) vector.
    initial_filters : int
        Number of filters in the first convolutional layer (ResNet stem).
    layers : List[int]
        Number of BasicBlock modules in each ResNet stage.

    Returns
    -------
    Tensor
        Reconstructed image of shape (batch_size, 3, height, width).
    """
    def __init__(
        self,
        height: int,
        width: int,
        latent_dim: int,
        init_filters: int = 64,
        layers: List[int] = [2, 2, 2, 2]
    ):
        super(ResNetAutoencoder, self).__init__()
        self.height = height
        self.width = width

        # Encoder
        self.conv1 = nn.Conv2d(3, init_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(init_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layers = nn.ModuleList()
        in_channels = init_filters
        for i, num_blocks in enumerate(layers):
            out_channels = in_channels if i == 0 else in_channels * 2
            stride = 1 if i == 0 else 2
            self.encoder_layers.append(self._make_layer(in_channels, out_channels, num_blocks, stride))
            in_channels = out_channels

        # Record encoder output shape
        with torch.no_grad():
            encoder_output_shape = self._forward_shapes(torch.zeros(1, 3, height, width))
        
        self._shapes = encoder_output_shape[::-1]
        self._shapes.append((height, width))

        # Bottleneck
        self.compress = nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False)
        self.decompress = nn.Conv2d(latent_dim, in_channels, kernel_size=1, bias=False)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        cur_channels = in_channels
        for i, (height_in, width_in) in enumerate(self._shapes[:-1]):
            height_out, width_out = self._shapes[i + 1]

            # Choose parameters
            if i == len(self._shapes) - 2:
                kernel_sizes, stride, padding = 7, 2, 3
                base_height = _deconv_out(height_in, kernel_sizes, stride, padding, 0)
                base_width = _deconv_out(width_in, kernel_sizes, stride, padding, 0)
                output_padding = (height_out - base_height, width_out - base_width)
            else:
                kernel_sizes, stride = 2, 2
                padding_height, output_padding_height = self._solve_dim(height_in, height_out, kernel_sizes, stride)
                padding_width, output_padding_width = self._solve_dim(width_in, width_out, kernel_sizes, stride)
                padding = (padding_height, padding_width)
                output_padding = (output_padding_height, output_padding_width)

            next_channels = max(init_filters, cur_channels // 2) if i < (len(self._shapes) - 2) else 3
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        cur_channels,
                        next_channels,
                        kernel_size=kernel_sizes,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False
                    ),
                    nn.BatchNorm2d(next_channels) if next_channels != 3 else nn.Identity(),
                    nn.ReLU(inplace=True) if next_channels != 3 else nn.Identity()
                )
            )
            cur_channels = next_channels

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        encoder_layers = [BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            encoder_layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*encoder_layers)
    
    def _forward_shapes(self, x: Tensor) -> List[Tuple[int, int]]:
        shapes = []
        x = self.conv1(x)
        shapes.append(x.shape[-2:])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        shapes.append(x.shape[-2:])

        for layer in self.encoder_layers[1:]:
            x = layer(x)
            shapes.append(x.shape[-2:])

        return shapes
    
    def _solve_dim(self, in_size: int, out_size: int, k: int = 2, s: int = 2) -> Tuple[int, int]:
        for p in (0, 1):
            base = _deconv_out(in_size, k, s, p, 0)
            diff = out_size - base
            if diff in (0, 1):
                return p, diff
            
        raise ValueError("No valid (padding, output_padding) found")

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.encoder_layers:
            x = layer(x)

        # Bottleneck
        x = self.compress(x)
        x = self.decompress(x)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        # Validate shape
        assert x.shape == (B, C, H, W), f"Output shape mismatch: {x.shape} != {(B, C, H, W)}"

        return x