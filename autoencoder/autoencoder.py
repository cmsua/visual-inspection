# Import necessary libraries
import os

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data_loading import *
from training import *

# Set the seed
torch.manual_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to the datasets folder
DATASET_PATH = './datasets'

# Path to the checkpoints folder
# CHECKPOINT_PATH = '/content/drive/MyDrive/Colab Notebooks/HGCAL/checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = Image.open(os.path.join(DATASET_PATH, 'HexaBoardExample.png'))

# Get the height and width of the image
width, height = image.size
print('Image width:', width)
print('Image height:', height)

# Adjust the number of segments
# THIS SHOULD WORK WITH THE GUI
NUM_VERTICAL_SEGMENTS = 20
NUM_HORIZONTAL_SEGMENTS = 12

# Define the transformations
transform = transforms.Compose([
    RotationAndSegmentationTransform(
        height=height,
        width=width,
        vertical_segments=NUM_VERTICAL_SEGMENTS,
        horizontal_segments=NUM_HORIZONTAL_SEGMENTS
    ),
    transforms.RandomRotation(degrees=2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

# Read in and process the iamges
dataset = HexaboardDataset(
    image_dir=os.path.join(DATASET_PATH),
    transform=transform
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
print(f'Segments Shape: {train_dataset[0].shape}')
print(f'Image shape: {train_dataset[0][0].shape}')
print(f'Image tensor type: {train_dataset[0][0].dtype}')
print(f'Batches: {len(train_loader)}')

height_post_transform = train_dataset[0][0][0].shape[0]
width_post_transform = train_dataset[0][0][0].shape[1]
print('Segment height:', height_post_transform)
print('Segment width:', width_post_transform)

chunk_size = 12

# Initialize the model
cnn_ae = ConvAutoEncoder(
    height=height_post_transform,
    width=width_post_transform,
    latent_dim=128,
    kernel_sizes=[32, 64]
)

cnn_ae.to(device)

# Load the model's weights
cnn_ae.load_state_dict(torch.load(r'C:\Users\thanh\Python\HGCAL\visual-inspection\autoencoder\small_ae.pt', map_location=device))

# Evaluate the model
criterion = nn.BCEWithLogitsLoss()

evaluate_autoencoder(
    model=cnn_ae,
    criterion=criterion,
    test_loader=test_loader,
    num_images=10,
    chunk_size=chunk_size,
    visualize=True
)