import torch
import torch.nn as nn
from typing import List

class SimpleCNNAutoEncoder(nn.Module):
    """
    A simple CNN-based Autoencoder with a bottleneck layer for dimensionality reduction.

    Args:
        height (int): The height of the input image.
        width (int): The width of the input image.
        latent_dim (int): The size of the latent space (bottleneck layer).
        kernel_sizes (List[int]): List of output channels for each Conv2D layer in the encoder.

    Returns:
        x (torch.Tensor): Reconstructed image tensor after encoding and decoding.
    """
    def __init__(self, height: int, width: int, latent_dim: int, kernel_sizes: List[int]) -> None:
        super(SimpleCNNAutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.kernel_sizes = kernel_sizes

        # Encoder architecture
        self.encoder = nn.ModuleList()
        self.padding = nn.ZeroPad2d((0, 0, 0, 3))  # Example padding, adjust based on needs

        for in_channels, out_channels in zip([3] + kernel_sizes[:-1], kernel_sizes):
            self.encoder.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2))
            self.encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.encoder)

        # Test input to determine the output shape after the encoder
        example_input = torch.randn(1, 3, height, width)
        output = self.encoder(self.padding(example_input))
        self.shape = output.shape  # Store the encoder output shape for later use in decoder

        # Bottleneck (latent space)
        self.bottleneck = nn.Linear(kernel_sizes[-1] * self.shape[2] * self.shape[3], latent_dim)
        self.unbottleneck = nn.Linear(latent_dim, kernel_sizes[-1] * self.shape[2] * self.shape[3])

        # Decoder architecture
        self.decoder = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(zip(kernel_sizes[::-1], kernel_sizes[:-1][::-1] + [3])):
            self.decoder.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2))
            if i < len(kernel_sizes) - 1:  # Apply ReLU except for the last layer
                self.decoder.append(nn.ReLU())

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            x (torch.Tensor): Reconstructed image tensor after encoding and decoding.
        """
        batch_size = x.size(0)

        # Apply padding and pass through the encoder
        x = self.padding(x)
        x = self.encoder(x)

        # Flatten and pass through the bottleneck
        x = x.view(batch_size, -1)
        x = self.bottleneck(x)

        # Unflatten and pass through the decoder
        x = self.unbottleneck(x)
        x = x.view(batch_size, self.kernel_sizes[-1], self.shape[2], self.shape[3])

        # Pass through the decoder
        x = self.decoder(x)

        # Remove the padding applied earlier
        x = x[:, :, :-1, :]

        return x