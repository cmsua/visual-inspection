# Import necessary libraries
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

# Define the custom transformation for rotations and segmentations
class RotationAndSegmentationTransform:
    def __init__(self, height, width, vertical_segments, horizontal_segments):
        self.height = height
        self.width = width
        self.vertical_segments = vertical_segments
        self.horizontal_segments = horizontal_segments

    def __call__(self, image):
        width = self.width
        a = width / 2
        height = int(np.sqrt(3) * a)

        segments = []
        segments.extend(get_segments(image, height, width, self.vertical_segments, self.horizontal_segments))
        segments.extend(get_segments(image, height, width, self.vertical_segments, self.horizontal_segments, 60))
        segments.extend(get_segments(image, height, width, self.vertical_segments, self.horizontal_segments, 240))

        # Convert PIL segments to tensors
        segments = [transforms.ToTensor()(segment) for segment in segments]

        segments = torch.stack(segments)

        return segments

# Define the custom dataset
class HexaboardDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_paths)
        img_path = self.image_paths[actual_idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

class ConvAutoEncoder(nn.Module):
    def __init__(self, height, width, latent_dim, kernel_sizes):
        super(ConvAutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.kernel_sizes = kernel_sizes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, kernel_sizes[0], kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(kernel_sizes[0], kernel_sizes[1], kernel_size=3, stride=2),
            nn.ReLU(True),
        )

        example_input = torch.randn(1, 3, height, width)
        output = self.encoder(example_input)
        self.shape = output.shape

        # Flatten to go into latent space
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(kernel_sizes[-1] * self.shape[2] * self.shape[3], latent_dim)

        # Latent space
        self.fc2 = nn.Linear(latent_dim, kernel_sizes[-1] * self.shape[2] * self.shape[3])
        self.unflatten = nn.Unflatten(1, (kernel_sizes[-1], self.shape[2], self.shape[3]))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kernel_sizes[1], kernel_sizes[0], kernel_size=3, stride=2, output_padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(kernel_sizes[0], 3, kernel_size=3, stride=2),
            # nn.Sigmoid(),  # For image outputs
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)

        # Decoding
        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x