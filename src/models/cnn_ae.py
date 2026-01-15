from typing import List, Optional, Tuple

from torch import nn, Tensor

from ..configs import AutoencoderConfig


class CNNAutoencoder(nn.Module):
    """
    A CNN autoencoder using ConvTranspose2d.
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
        
        # Use config if provided, otherwise use defaults
        if config is not None:
            self.height = height if height is not None else config.height
            self.width = width if width is not None else config.width
            self.latent_dim = latent_dim if latent_dim is not None else config.latent_dim
            self.init_filters = init_filters if init_filters is not None else config.init_filters
            self.layers = layers if layers is not None else config.layers
        else:
            self.height = height if height is not None else 1080
            self.width = width if width is not None else 1920
            self.latent_dim = latent_dim if latent_dim is not None else 32
            self.init_filters = init_filters if init_filters is not None else 128
            self.layers = layers if layers is not None else [2, 2, 2]

        # Bottleneck size
        self.bottleneck_height = 9
        self.bottleneck_width = 16

        # Use GroupNorm
        self.norm_layer = lambda num_channels: nn.GroupNorm(1, num_channels)

        # Encoder stem (downsample x8)
        self.conv1 = nn.Conv2d(3, self.init_filters, kernel_size=4, stride=4, padding=0, bias=True)
        self.bn1 = self.norm_layer(self.init_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        stem_out_height = self._conv_out_size(self.height, 4, 4, 0)
        stem_out_width = self._conv_out_size(self.width, 4, 4, 0)
        stem_out_height = self._conv_out_size(stem_out_height, 2, 2, 0)
        stem_out_width = self._conv_out_size(stem_out_width, 2, 2, 0)

        if stem_out_height % self.bottleneck_height != 0 or stem_out_width % self.bottleneck_width != 0:
            raise ValueError("Stem output must be divisible by bottleneck size.")

        remaining_scale_h = stem_out_height // self.bottleneck_height
        remaining_scale_w = stem_out_width // self.bottleneck_width
        if remaining_scale_h != remaining_scale_w:
            raise ValueError("Height/width downsample scales must match.")

        stage_strides = self._split_scale(remaining_scale_h, len(self.layers) - 1)
        self.stage_strides = [1] + stage_strides

        self.encoder_layers = nn.ModuleList()
        in_channels = self.init_filters
        for i, num_blocks in enumerate(self.layers):
            if i == 0:
                out_channels = in_channels
            elif i == len(self.layers) - 1:
                out_channels = self.latent_dim
            else:
                out_channels = in_channels * 2

            stride = self.stage_strides[i]
            stage = self._make_stage(in_channels, out_channels, num_blocks, stride)
            self.encoder_layers.append(stage)
            in_channels = out_channels

        # Encoder output per stage
        encoder_channels = [self.init_filters]
        for i in range(1, len(self.layers) - 1):
            encoder_channels.append(encoder_channels[-1] * 2)

        encoder_channels.append(self.latent_dim)

        bottleneck_height = stem_out_height
        bottleneck_width = stem_out_width
        for stride in self.stage_strides:
            if stride > 1:
                bottleneck_height = self._conv_out_size(bottleneck_height, stride, stride, 0)
                bottleneck_width = self._conv_out_size(bottleneck_width, stride, stride, 0)

        if bottleneck_height != self.bottleneck_height or bottleneck_width != self.bottleneck_width:
            raise ValueError("Encoder does not reach the configured bottleneck size.")

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(self.layers) - 1, -1, -1):
            in_channels = encoder_channels[i - 1] if i > 0 else encoder_channels[0]
            out_channels = encoder_channels[i]
            blocks = self.layers[i]
            stride = self.stage_strides[i]

            for _ in range(blocks - 1):
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True
                        ),
                        self.norm_layer(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )

            if stride == 1:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = stride
                padding = 0

            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=out_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=True
                    ),
                    self.norm_layer(in_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Undo stem maxpool
        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=encoder_channels[0],
                    out_channels=encoder_channels[0],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True
                ),
                self.norm_layer(encoder_channels[0]),
                nn.ReLU(inplace=True),
            )
        )

        # Invert the stem to original size
        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=encoder_channels[0],
                    out_channels=3,
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True
                )
            )
        )

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []

        # First block (may downsample via stride)
        if stride == 1:
            kernel_size = 3
            padding = 1
        else:
            kernel_size = stride
            padding = 0

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True
                ),
                self.norm_layer(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Remaining blocks (stride=1)
        for _ in range(1, blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    ),
                    self.norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        return nn.Sequential(*layers)

    @staticmethod
    def _conv_out_size(size: int, kernel: int, stride: int, padding: int) -> int:
        return (size + 2 * padding - kernel) // stride + 1

    @staticmethod
    def _split_scale(scale: int, parts: int) -> List[int]:
        if parts <= 0:
            raise ValueError("Stage count must be positive.")
        if parts == 1:
            return [scale]

        factors = []
        remainder = scale
        for prime in (5, 3, 2):
            while remainder % prime == 0 and len(factors) < parts:
                factors.append(prime)
                remainder //= prime

        if remainder != 1:
            factors.append(remainder)

        while len(factors) < parts:
            factors.append(1)

        while len(factors) > parts:
            factors[-2] *= factors[-1]
            factors.pop()

        return factors

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

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.encoder_layers:
            x = layer(x)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        # Validate shape
        assert x.shape == (B, C, H, W), f"Output shape mismatch: {x.shape} != {(B, C, H, W)}"

        return x