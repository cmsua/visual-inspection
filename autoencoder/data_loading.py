# Import necessary libraries
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

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
class SimpleCNNAutoEncoder(nn.Module):
    def __init__(self, height, width, latent_dim, kernel_sizes):
        super(SimpleCNNAutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.kernel_sizes = kernel_sizes
        self.encoder = nn.ModuleList()

        # Padding to make dimensions square or divisible by the necessary factor
        self.padding = nn.ZeroPad2d((0, 0, 0, 3))  # Example padding, adjust as needed

        for i, (indim, outdim) in enumerate(zip([3] + kernel_sizes[:-1], kernel_sizes)):
            layer = nn.Conv2d(indim, outdim, kernel_size=3, stride=2)
            self.encoder.append(layer)
            self.encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.encoder)

        example_input = torch.randn(1, 3, height, width)
        # output = self.encoder(example_input)
        output = self.encoder(self.padding(example_input))
        self.shape = output.shape

        self.bottleneck = nn.Linear(kernel_sizes[-1] * self.shape[2] * self.shape[3], latent_dim)
        self.unbottleneck = nn.Linear(latent_dim, kernel_sizes[-1] * self.shape[2] * self.shape[3])

        self.decoder = nn.ModuleList()

        for i, (indim, outdim) in enumerate(zip(kernel_sizes[::-1], kernel_sizes[:-1][::-1] + [3])):
            if i == len(kernel_sizes) - 2:
                layer = nn.ConvTranspose2d(indim, outdim, kernel_size=3, stride=2)
            else:
                layer = nn.ConvTranspose2d(indim, outdim, kernel_size=3, stride=2)
            self.decoder.append(layer)
            if i < len(kernel_sizes) - 1:
                self.decoder.append(nn.ReLU())

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.padding(x)  # Apply padding
        x = self.encoder(x)
        x = x.view(batch, -1)
        x = self.bottleneck(x)
        x = self.unbottleneck(x)
        x = x.view(batch, self.kernel_sizes[-1], self.shape[2], self.shape[3])
        x = self.decoder(x)
        x = x[:, :, :-1, :]  # Remove the padding from the width
        return x