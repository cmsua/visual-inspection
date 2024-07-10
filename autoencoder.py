# Import necessary libraries
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

matplotlib.rcParams['lines.linewidth'] = 2.0

# Set the seed
torch.manual_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get the current working directory
current_dir = os.getcwd()

# Add the project directory to the system path
sys.path.append(current_dir)

# Path to the datasets folder
DATASET_PATH = os.path.join(current_dir, 'datasets')

# Path to the checkpoints folder
# CHECKPOINT_PATH = '/content/drive/MyDrive/Colab Notebooks/HGCAL/checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_segments_1(image, height, width, vertical_segments, horizontal_segments):
    a = width / 2
    transform = transforms.Compose([
        transforms.CenterCrop((int(height), int(width))),
        transforms.CenterCrop((int(height), int(a)))
    ])

    cropped_image = transform(image)

    # Calculate the dimensions of each segment
    segment_height = int(height / vertical_segments)
    segment_width = int(a / horizontal_segments)

    segments = []

    # Split the cropped image into subsegments
    for i in range(vertical_segments):
        for j in range(horizontal_segments):
            left = j * segment_width
            upper = i * segment_height
            right = left + segment_width
            lower = upper + segment_height

            segment = cropped_image.crop((left, upper, right, lower))
            segments.append(segment)

    return segments


def get_segments_2(image, height, width, vertical_segments, horizontal_segments):
    a = width / 2
    transform = transforms.Compose([
        transforms.CenterCrop((int(height), int(width))),
        transforms.RandomRotation(degrees=(60, 60)),
        transforms.Resize((height, width)),
        transforms.CenterCrop((int(height), int(a)))
    ])

    cropped_image = transform(image)

    # Calculate the dimensions of each segment
    segment_height = int(height / vertical_segments)
    segment_width = int(a / horizontal_segments)

    segments = []

    # Split the cropped image into subsegments
    for i in range(vertical_segments):
        for j in range(horizontal_segments):
            left = j * segment_width
            upper = i * segment_height
            right = left + segment_width
            lower = upper + segment_height

            segment = cropped_image.crop((left, upper, right, lower))
            segments.append(segment)

    return segments

def get_segments_3(image, height, width, vertical_segments, horizontal_segments):
    a = width / 2
    transform = transforms.Compose([
        transforms.CenterCrop((int(height), int(width))),
        transforms.RandomRotation(degrees=(240, 240)),
        transforms.Resize((height, width)),
        transforms.CenterCrop((int(height), int(a)))
    ])

    cropped_image = transform(image)

    # Calculate the dimensions of each segment
    segment_height = int(height / vertical_segments)
    segment_width = int(a / horizontal_segments)

    segments = []

    # Split the cropped image into subsegments
    for i in range(vertical_segments):
        for j in range(horizontal_segments):
            left = j * segment_width
            upper = i * segment_height
            right = left + segment_width
            lower = upper + segment_height

            segment = cropped_image.crop((left, upper, right, lower))
            segments.append(segment)

    return segments

image = Image.open(os.path.join(DATASET_PATH, 'HexaBoardExample.png'))

# Get the height and width of the image
width, height = image.size
print('Image width:', width)
print('Image height:', height)

# Adjust the number of segments
# THIS SHOULD WORK WITH THE GUI
NUM_VERTICAL_SEGMENTS = 20
NUM_HORIZONTAL_SEGMENTS = 12

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
        segments.extend(get_segments_1(image, height, width, self.vertical_segments, self.horizontal_segments))
        segments.extend(get_segments_2(image, height, width, self.vertical_segments, self.horizontal_segments))
        segments.extend(get_segments_3(image, height, width, self.vertical_segments, self.horizontal_segments))

        # Convert PIL segments to tensors
        segments = [transforms.ToTensor()(segment) for segment in segments]

        segments = torch.stack(segments)

        return segments

# Define the custom dataset
class HexaboardDataset(Dataset):
    def __init__(self, image_dir, transform=None, num_augmentations=1000):
        self.image_dir = image_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_paths)
        img_path = self.image_paths[actual_idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define the transformations
transform = transforms.Compose([
    RotationAndSegmentationTransform(
        height=height,
        width=width,
        vertical_segments=NUM_VERTICAL_SEGMENTS,
        horizontal_segments=NUM_HORIZONTAL_SEGMENTS
    ),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

# Read in and process the iamges
dataset = HexaboardDataset(
    image_dir=os.path.join(DATASET_PATH),
    transform=transform,
    num_augmentations=1000
)

# Split the dataset to get train, validation, and test sets
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
val_dataset, test_dataset = random_split(val_dataset, [0.5, 0.5])

# Set the batch size
batch_size = 1

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Print some information about the data
print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')
print(f'Segements Shape: {train_dataset[0].shape}')
print(f'Image shape: {train_dataset[0][0].shape}')
print(f'Image tensor type: {train_dataset[0][0].dtype}')
print(f'Batches: {len(train_loader)}')

height_post_transform = train_dataset[0][0][0].shape[0]
width_post_transform = train_dataset[0][0][0].shape[1]
print('Segment height:', height_post_transform)
print('Segment width:', width_post_transform)

chunk_size = 12

def visualize_reconstructions(original_images, reconstructed_images, num_images, save_fig):
    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        img = original_images[i][0].permute(1, 2, 0).numpy()  # handle batch dimension
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        if i == 0:
            plt.title('Original')

        # Reconstructed images
        plt.subplot(2, num_images, i + 1 + num_images)
        img = reconstructed_images[i][0].permute(1, 2, 0).numpy()  # handle batch dimension
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')

    if save_fig:
        plt.savefig(os.path.join(current_dir, 'reconstructions.png'))

    plt.show()

def evaluate_autoencoder(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: DataLoader,
    num_images: int = 8,
    save_fig: bool = False
):
    model.eval()
    total_loss = 0.0

    original_images = []
    reconstructed_images = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            inputs = inputs.squeeze(0)
            for input in inputs.chunk(chunk_size):
                intput = input.squeeze(0)
                output = model(input)
                loss = criterion(output, input)

                total_loss += loss.item()

                # Save some images for visualization
                if len(original_images) < num_images:
                    original_images.append(input.cpu())
                    reconstructed_images.append(torch.sigmoid(output.cpu()))

    test_loss = total_loss / chunk_size / len(test_loader)

    print(f'Test Loss: {test_loss:.4f}')

    # Visualize the results
    visualize_reconstructions(original_images, reconstructed_images, num_images, save_fig)

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
        self.unflatten = nn.Unflatten(1,  (kernel_sizes[-1], self.shape[2], self.shape[3]))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kernel_sizes[1], kernel_sizes[0], kernel_size=3, stride=2, output_padding=(0,1)),
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

# Initialize the model
cnn_ae = ConvAutoEncoder(
    height=height_post_transform,
    width=width_post_transform,
    latent_dim=128,
    kernel_sizes=[32, 64]
)

cnn_ae.to(device)

# Load the model's weights
cnn_ae.load_state_dict(torch.load(os.path.join(current_dir, 'small_ae.pt'), map_location=device))

# Evaluate the model
criterion = nn.BCEWithLogitsLoss()

evaluate_autoencoder(
    model=cnn_ae,
    criterion=criterion,
    test_loader=test_loader,
    num_images=10,
    save_fig=False
)