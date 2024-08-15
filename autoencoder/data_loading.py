# Import necessary libraries
import os
import sys
from collections.abc import MutableSequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

# Directory path used in local
project_dir = './'
sys.path.append(project_dir)

from utils import get_segments

# Define the custom transformation for rotations and segmentations
class RotationAndSegmentationTransform:
    def __init__(self, height: int, width: int, vertical_segments: int, horizontal_segments: int, tf: transforms.Compose = None):
        self.height = height
        self.width = width
        self.vertical_segments = vertical_segments
        self.horizontal_segments = horizontal_segments

        if tf is None:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.tf = tf

    def __call__(self, image: Image) -> torch.Tensor:
        width = self.width
        a = width / 2
        height = int(np.sqrt(3) * a)

        segments = []
        segments.extend(get_segments(image, height, width, self.vertical_segments, self.horizontal_segments))
        segments.extend(get_segments(image, height, width, self.vertical_segments, self.horizontal_segments, 60))
        segments.extend(get_segments(image, height, width, self.vertical_segments, self.horizontal_segments, 300))

        # Convert PIL segments to tensors
        segments = [self.tf(segment) for segment in segments]

        segments = torch.stack(segments)

        return segments

# Define the custom dataset
class HexaboardDataset(Dataset):
    def __init__(self, image_dir: str, transform: RotationAndSegmentationTransform = RotationAndSegmentationTransform) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Image:
        actual_idx = idx % len(self.image_paths)
        img_path = self.image_paths[actual_idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# Model architecture
class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, latent_dim: int, kernel_sizes: MutableSequence[int]):
        super(ConvAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.kernel_sizes = kernel_sizes

        # Encoder
        encoder_layers = []

        for out_channels in self.kernel_sizes:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2))
            encoder_layers.append(nn.ReLU(True))
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

        example_input = torch.randn(1, self.in_channels, height, width)
        with torch.no_grad():
            output = self.encoder(example_input)
        self.shape = output.shape

        # Flatten to go into latent space
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.kernel_sizes[-1] * self.shape[2] * self.shape[3], latent_dim)

        # Latent space
        self.fc2 = nn.Linear(latent_dim, self.kernel_sizes[-1] * self.shape[2] * self.shape[3])
        self.unflatten = nn.Unflatten(1, (self.kernel_sizes[-1], self.shape[2], self.shape[3]))

        # Decoder
        decoder_layers = []

        for i, out_channels in enumerate(self.kernel_sizes[::-1]):
            in_channels = self.kernel_sizes[::-1][i - 1] if i > 0 else self.kernel_sizes[-1]
            if i == len(self.kernel_sizes) - 1:
                out_channels = self.in_channels
                decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2))
                break
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, output_padding=(0, 1)))
            decoder_layers.append(nn.ReLU(True))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)

        # Decoding
        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

class SimpleCNNAutoEncoder(nn.Module):
    def __init__(self, height, width, latent_dim, kernel_sizes):
        super(SimpleCNNAutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.kernel_sizes = kernel_sizes
        self.encoder = nn.ModuleList()

        for i, (indim, outdim) in enumerate(zip([3] + kernel_sizes[:-1], kernel_sizes)):
            layer = nn.Conv2d(indim, outdim, kernel_size=3, stride=2, padding=(outdim-3)%3)
            self.encoder.append(layer)
            self.encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.encoder)

        example_input = torch.randn(1, 3, height, width)
        output = self.encoder(example_input)
        self.shape = output.shape


        self.bottleneck = nn.Linear(kernel_sizes[-1] * self.shape[2] * self.shape[3], latent_dim)
        self.unbottleneck = nn.Linear(latent_dim, kernel_sizes[-1] * self.shape[2] * self.shape[3])

        self.decoder = nn.ModuleList()

        for i, (indim, outdim) in enumerate(zip(kernel_sizes[::-1], kernel_sizes[:-1][::-1] + [3])):
            if i == len(kernel_sizes) - 2:
                layer = nn.ConvTranspose2d(indim, outdim, kernel_size=3, stride=2, padding=(indim - 3) % 3, output_padding=(1, 0))
            else:
                layer = nn.ConvTranspose2d(indim, outdim, kernel_size=3, stride=2, padding=(indim - 3) % 3)
            self.decoder.append(layer)
            if i < len(kernel_sizes) - 1:
                self.decoder.append(nn.ReLU())

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.encoder(x)
        x = x.view(batch, -1)
        x = self.bottleneck(x)
        x = self.unbottleneck(x)
        x = x.view(batch, self.kernel_sizes[-1], self.shape[2], self.shape[3])
        x = self.decoder(x)
        return x