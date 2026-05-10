from typing import Dict, List, Optional, Tuple

from torch import Tensor, nn

from ..configs import AutoencoderConfig


class CNNAutoencoder(nn.Module):
    """
    A symmetric CNN autoencoder for segment-level anomaly detection.
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
            Configuration object containing model parameters.
        height: int, optional
            Height of the input images.
        width: int, optional
            Width of the input images.
        latent_dim: int, optional
            Dimension of the latent (bottleneck) vector.
        init_filters: int, optional
            Number of filters in the first convolutional layer (stem).
        layers: List[int], optional
            Number of Conv-GN-ReLU blocks in each encoder stage.
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

        if len(self.layers) != 3:
            raise ValueError('CNNAutoencoder expects exactly three encoder stages.')

        self.norm_layer = lambda num_channels: nn.GroupNorm(1, num_channels)

        self.stem_kernel = (8, 8)
        self.stem_stride = (4, 4)
        self.stem_padding = (2, 3)

        shapes = self._compute_shapes()
        self._shapes = [
            shapes['stem'],
            shapes['post_stem_pool'],
            shapes['stage1_conv'],
            shapes['stage1_postpool'],
            shapes['stage2_conv'],
            shapes['stage2_postpool'],
        ]

        self.bottleneck_height, self.bottleneck_width = shapes['stage2_postpool']
        self.bottleneck_area = self.bottleneck_height * self.bottleneck_width

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.init_filters,
            kernel_size=self.stem_kernel,
            stride=self.stem_stride,
            padding=self.stem_padding,
            bias=True,
        )
        self.bn1 = self.norm_layer(self.init_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        encoder_channels = [self.init_filters, self.init_filters * 2, self.latent_dim]

        self.encoder_layers = nn.ModuleList(
            [
                self._make_stage(
                    in_channels=self.init_filters,
                    out_channels=encoder_channels[0],
                    blocks=self.layers[0],
                    stride=1,
                    pooling=False,
                ),
                self._make_stage(
                    in_channels=encoder_channels[0],
                    out_channels=encoder_channels[1],
                    blocks=self.layers[1],
                    stride=2,
                    pooling=True,
                    pool_padding=0,
                ),
                self._make_stage(
                    in_channels=encoder_channels[1],
                    out_channels=encoder_channels[2],
                    blocks=self.layers[2],
                    stride=2,
                    pooling=True,
                    pool_padding=1,
                ),
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                self._make_decoder_layer(
                    in_channels=encoder_channels[2],
                    out_channels=encoder_channels[1],
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(1, 1),
                    input_size=shapes['stage2_postpool'],
                    target_size=shapes['stage2_conv'],
                ),
                self._make_decoder_layer(
                    in_channels=encoder_channels[1],
                    out_channels=encoder_channels[1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    input_size=shapes['stage2_conv'],
                    target_size=shapes['stage1_postpool'],
                ),
                self._make_decoder_layer(
                    in_channels=encoder_channels[1],
                    out_channels=encoder_channels[0],
                    kernel_size=(3, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    input_size=shapes['stage1_postpool'],
                    target_size=shapes['stage1_conv'],
                ),
                self._make_decoder_layer(
                    in_channels=encoder_channels[0],
                    out_channels=encoder_channels[0],
                    kernel_size=(3, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    input_size=shapes['stage1_conv'],
                    target_size=shapes['post_stem_pool'],
                ),
                self._make_decoder_layer(
                    in_channels=encoder_channels[0],
                    out_channels=encoder_channels[0],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    input_size=shapes['post_stem_pool'],
                    target_size=shapes['stem'],
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=encoder_channels[0],
                        out_channels=3,
                        kernel_size=self.stem_kernel,
                        stride=self.stem_stride,
                        padding=self.stem_padding,
                        output_padding=self._compute_output_padding(
                            input_size=shapes['stem'],
                            target_size=(self.height, self.width),
                            kernel_size=self.stem_kernel,
                            stride=self.stem_stride,
                            padding=self.stem_padding,
                        ),
                        bias=True,
                    )
                ),
            ]
        )

    @staticmethod
    def _conv_out_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
        return ((size + (2 * padding) - kernel_size) // stride) + 1

    @staticmethod
    def _pool_out_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
        return ((size + (2 * padding) - kernel_size) // stride) + 1

    @classmethod
    def _compute_output_padding(
        cls,
        input_size: Tuple[int, int],
        target_size: Tuple[int, int],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Tuple[int, int]:
        base_height = ((input_size[0] - 1) * stride[0]) - (2 * padding[0]) + kernel_size[0]
        base_width = ((input_size[1] - 1) * stride[1]) - (2 * padding[1]) + kernel_size[1]
        output_padding = (target_size[0] - base_height, target_size[1] - base_width)
        if any(value not in (0, 1) for value in output_padding):
            raise ValueError(
                f'Invalid output padding from {input_size} to {target_size}: {output_padding}'
            )

        return output_padding

    def _compute_shapes(self) -> Dict[str, Tuple[int, int]]:
        stem_height = self._conv_out_size(
            self.height,
            kernel_size=self.stem_kernel[0],
            stride=self.stem_stride[0],
            padding=self.stem_padding[0],
        )
        stem_width = self._conv_out_size(
            self.width,
            kernel_size=self.stem_kernel[1],
            stride=self.stem_stride[1],
            padding=self.stem_padding[1],
        )

        post_stem_pool = (
            self._pool_out_size(stem_height, kernel_size=3, stride=2, padding=1),
            self._pool_out_size(stem_width, kernel_size=3, stride=2, padding=1),
        )
        stage1_conv = (
            self._conv_out_size(post_stem_pool[0], kernel_size=3, stride=2, padding=1),
            self._conv_out_size(post_stem_pool[1], kernel_size=3, stride=2, padding=1),
        )
        stage1_postpool = (
            self._pool_out_size(stage1_conv[0], kernel_size=2, stride=2, padding=0),
            self._pool_out_size(stage1_conv[1], kernel_size=2, stride=2, padding=0),
        )
        stage2_conv = (
            self._conv_out_size(stage1_postpool[0], kernel_size=3, stride=2, padding=1),
            self._conv_out_size(stage1_postpool[1], kernel_size=3, stride=2, padding=1),
        )
        stage2_postpool = (
            self._pool_out_size(stage2_conv[0], kernel_size=2, stride=2, padding=1),
            self._pool_out_size(stage2_conv[1], kernel_size=2, stride=2, padding=1),
        )

        return {
            'stem': (stem_height, stem_width),
            'post_stem_pool': post_stem_pool,
            'stage1_conv': stage1_conv,
            'stage1_postpool': stage1_postpool,
            'stage2_conv': stage2_conv,
            'stage2_postpool': stage2_postpool,
        }

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        pooling: bool = False,
        pool_padding: int = 0,
    ) -> nn.Sequential:
        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=True,
                ),
                self.norm_layer(out_channels),
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
                    self.norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=pool_padding))

        return nn.Sequential(*layers)

    def _make_decoder_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        input_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=self._compute_output_padding(
                    input_size=input_size,
                    target_size=target_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                bias=True,
            ),
            self.norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def _forward_shapes(self, x: Tensor) -> List[Tuple[int, int]]:
        shapes = []
        x = self.conv1(x)
        shapes.append(x.shape[-2:])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        shapes.append(x.shape[-2:])

        for layer in self.encoder_layers:
            x = layer(x)
            shapes.append(x.shape[-2:])

        return shapes

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        return self._shapes

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.encoder_layers:
            x = layer(x)

        for layer in self.decoder_layers:
            x = layer(x)

        assert x.shape == (batch_size, channels, height, width), (
            f'Output shape mismatch: {x.shape} != {(batch_size, channels, height, width)}'
        )

        return x
