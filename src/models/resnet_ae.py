import math
from typing import List, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock

from ..configs import AutoencoderConfig


class ResNetAutoencoder(nn.Module):
    """
    A ResNet-style autoencoder for segment-level anomaly detection.
    """
    def __init__(
        self,
        config: Optional[AutoencoderConfig] = None,
        # Parameters below can override config if supplied explicitly
        height: Optional[int] = None,
        width: Optional[int] = None,
        latent_dim: Optional[int] = None,
        init_filters: Optional[int] = None,
        layers: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        config: AutoencoderConfig, optional
            Configuration object containing model parameters. If provided, it will override individual parameters.
        height: int, optional
            Input image height. Ignored if `config` is provided.
        width: int, optional
            Input image width. Ignored if `config` is provided.
        latent_dim: int, optional
            Dimensionality of the latent space. Ignored if `config` is provided.
        init_filters: int, optional
            Number of filters in the first convolutional layer. Ignored if `config` is provided.
        layers: List[int], optional
            List specifying the number of blocks in each encoder stage. Ignored if `config` is provided.
        """
        super().__init__()

        if config is not None:
            self.height = height if height is not None else config.height
            self.width = width if width is not None else config.width
            self.latent_dim = latent_dim if latent_dim is not None else config.latent_dim
            self.init_filters = init_filters if init_filters is not None else config.init_filters
            self.layers = layers if layers is not None else config.layers
        else:
            self.height = height if height is not None else 1060
            self.width = width if width is not None else 1882
            self.latent_dim = latent_dim if latent_dim is not None else 32
            self.init_filters = init_filters if init_filters is not None else 128
            self.layers = layers if layers is not None else [2, 2, 2]

        if not self.layers:
            raise ValueError("`layers` must contain at least one stage.")

        self.bottleneck_height, self.bottleneck_width = self._choose_bottleneck_shape(
            self.height,
            self.width,
            max_area=160,
        )
        self.bottleneck_area = self.bottleneck_height * self.bottleneck_width

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.init_filters, kernel_size=7, stride=2, padding=3, bias=True),
            self._norm(self.init_filters),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        encoder_stage_channels = self._build_stage_channels()
        self.encoder_stage_channels = encoder_stage_channels

        self.encoder_layers = nn.ModuleList()
        in_channels = self.init_filters
        for stage_idx, num_blocks in enumerate(self.layers):
            stride = 1 if stage_idx == 0 else 2
            out_channels = encoder_stage_channels[stage_idx]
            stage = self._make_encoder_stage(in_channels, out_channels, num_blocks, stride)
            self.encoder_layers.append(stage)
            in_channels = out_channels

        self.bottleneck_pool = nn.AdaptiveAvgPool2d((self.bottleneck_height, self.bottleneck_width))
        self.bottleneck_reduce = nn.Sequential(
            nn.Conv2d(encoder_stage_channels[-1], self.latent_dim, kernel_size=1, bias=True),
            self._norm(self.latent_dim),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_expand = nn.Sequential(
            nn.Conv2d(self.latent_dim, encoder_stage_channels[-1], kernel_size=1, bias=True),
            self._norm(encoder_stage_channels[-1]),
            nn.ReLU(inplace=True),
        )

        reversed_channels = list(reversed(encoder_stage_channels))
        reversed_blocks = list(reversed(self.layers))
        self.decoder_layers = nn.ModuleList()
        current_channels = encoder_stage_channels[-1]
        for out_channels, num_blocks in zip(reversed_channels, reversed_blocks):
            stage = self._make_decoder_stage(current_channels, out_channels, num_blocks)
            self.decoder_layers.append(stage)
            current_channels = out_channels

        self.stem_decoder = self._make_decoder_stage(current_channels, self.init_filters, 1)
        self.output_head = nn.Conv2d(self.init_filters, 3, kernel_size=3, stride=1, padding=1, bias=True)

    @staticmethod
    def _norm(num_channels: int) -> nn.GroupNorm:
        return nn.GroupNorm(1, num_channels)

    def _build_stage_channels(self) -> List[int]:
        channels = []
        current_channels = self.init_filters
        for stage_idx in range(len(self.layers)):
            if 0 < stage_idx < len(self.layers) - 1:
                current_channels *= 2
            channels.append(current_channels)

        return channels

    def _make_encoder_stage(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                self._norm(out_channels),
            )

        layers.append(
            BasicBlock(
                inplanes=in_channels,
                planes=out_channels,
                stride=stride,
                downsample=downsample,
                norm_layer=self._norm,
            )
        )

        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, norm_layer=self._norm))

        return nn.Sequential(*layers)

    def _make_decoder_stage(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
    ) -> nn.Sequential:
        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                self._norm(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for _ in range(1, blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    self._norm(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        return nn.Sequential(*layers)

    @staticmethod
    def _choose_bottleneck_shape(height: int, width: int, max_area: int) -> Tuple[int, int]:
        aspect_ratio = height / width
        best_candidate = None

        for bottleneck_height in range(2, max_area + 1):
            max_width = max_area // bottleneck_height
            for bottleneck_width in range(2, max_width + 1):
                candidate_ratio = bottleneck_height / bottleneck_width
                ratio_error = abs(math.log(candidate_ratio / aspect_ratio))
                area = bottleneck_height * bottleneck_width
                candidate = (ratio_error, -area, bottleneck_height, bottleneck_width)
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

        if best_candidate is None:
            raise ValueError("Could not determine a valid bottleneck shape.")

        _, _, bottleneck_height, bottleneck_width = best_candidate
        return bottleneck_height, bottleneck_width

    def _forward_shapes(self, x: Tensor) -> List[Tuple[int, int]]:
        shapes = []
        x = self.stem(x)
        shapes.append(x.shape[-2:])
        x = self.pool(x)
        shapes.append(x.shape[-2:])
        for layer in self.encoder_layers:
            x = layer(x)
            shapes.append(x.shape[-2:])

        x = self.bottleneck_pool(x)
        shapes.append(x.shape[-2:])
        return shapes

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape

        stage_sizes = []

        x = self.stem(x)
        stem_size = x.shape[-2:]
        x = self.pool(x)

        for layer in self.encoder_layers:
            x = layer(x)
            stage_sizes.append(x.shape[-2:])

        x = self.bottleneck_pool(x)
        x = self.bottleneck_reduce(x)
        x = self.bottleneck_expand(x)

        target_sizes = list(reversed(stage_sizes))
        for layer, target_size in zip(self.decoder_layers, target_sizes):
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = layer(x)

        x = F.interpolate(x, size=stem_size, mode='bilinear', align_corners=False)
        x = self.stem_decoder(x)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        x = self.output_head(x)

        assert x.shape == (batch_size, channels, height, width), (
            f"Output shape mismatch: {x.shape} != {(batch_size, channels, height, width)}"
        )

        return x
