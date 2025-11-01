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
        height : int, optional
            Height of the input images.
        width : int, optional
            Width of the input images.
        latent_dim : int, optional
            Dimension of the latent (bottleneck) vector.
        init_filters : int, optional
            Number of filters in the first convolutional layer (stem).
        layers : List[int], optional
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
            self.height = height if height is not None else 1016
            self.width = width if width is not None else 1640
            self.latent_dim = latent_dim if latent_dim is not None else 32
            self.init_filters = init_filters if init_filters is not None else 128
            self.layers = layers if layers is not None else [2, 2, 2]

        # Use GroupNorm
        self.norm_layer = lambda num_channels: nn.GroupNorm(1, num_channels)

        # Encoder
        self.conv1 = nn.Conv2d(3, self.init_filters, kernel_size=(10, 16), stride=(5, 8), padding=(5, 0), bias=True)
        self.bn1 = self.norm_layer(self.init_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layers = nn.ModuleList()
        in_channels = self.init_filters
        for i, num_blocks in enumerate(self.layers):
            if i == 0:
                out_channels = in_channels
            elif i == len(self.layers) - 1:
                out_channels = self.latent_dim
            else:
                out_channels = in_channels * 2

            stride = 1 if i == 0 else 2  # downsample at the start of each stage (except stage 0)
            pooling = False if i == 0 else True  # pooling only after the first stage
            stage = self._make_stage(in_channels, out_channels, num_blocks, stride, pooling)
            self.encoder_layers.append(stage)
            in_channels = out_channels

        # Sizes produced by the stem
        conv1_out_height = (self.height + 10 - 10) // 5 + 1
        conv1_out_width = (self.width - 16) // 8 + 1
        stem_square_size = (conv1_out_height + 2 - 3) // 2 + 1
        assert stem_square_size == ((conv1_out_width + 2 - 3) // 2 + 1), "Stem must yield a square map."

        # Encoder output per stage
        encoder_channels = [self.init_filters]
        for i in range(1, len(self.layers) - 1):
            encoder_channels.append(encoder_channels[-1] * 2)

        encoder_channels.append(self.latent_dim)

        # Spatial size per stage
        stage_out = [stem_square_size]
        stage_conv = [None] * len(self.layers)
        for i in range(1, len(self.layers)):
            stride_conv = (stage_out[i - 1] - 1) // 2 + 1
            stride_pool = stride_conv // 2
            stage_conv[i] = stride_conv
            stage_out.append(stride_pool)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(self.layers) - 1, 0, -1):
            in_channels = encoder_channels[i]
            mid_channels = encoder_channels[i - 1]

            # Undo pooling
            output_padding1 = stage_conv[i] - 2 * stage_out[i]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        mid_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=int(output_padding1),
                        bias=True
                    ),
                    self.norm_layer(mid_channels),
                    nn.ReLU(inplace=True),
                )
            )

            # Undo convolution layers with stride=2
            output_padding2 = stage_out[i - 1] - (2 * stage_conv[i] - 1)
            assert output_padding2 in (0, 1), f"invalid output_padding2={output_padding2} for stage {i}"
            
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        mid_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=int(output_padding2),
                        bias=True
                    ),
                    self.norm_layer(mid_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Undo stem maxpool
        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    encoder_channels[0],
                    encoder_channels[0],
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
        out_height = self.height - ((conv1_out_height - 1) * 5 - 10 + 10)
        out_width = self.width - ((conv1_out_width - 1) * 8 + 16)
        assert out_height in (0, 1) and out_width in (0, 1), "Stem inverse requires op in {0, 1}."

        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    encoder_channels[0],
                    3,
                    kernel_size=(10, 16),
                    stride=(5, 8),
                    padding=(5, 0),
                    output_padding=(int(out_height), int(out_width)),
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
        pooling: bool = False,
    ) -> nn.Sequential:
        layers = []

        # First block (may downsample via stride)
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            self.norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ))

        # Remaining blocks (stride=1)
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                self.norm_layer(out_channels),
                nn.ReLU(inplace=True),
            ))

        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

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