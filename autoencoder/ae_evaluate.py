# Import necessary dependencies
import os

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Define project directory
project_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
autoencoder_dir = os.path.join(project_dir, 'autoencoder')

# Import modules using absolute imports
from preprocessing.ae_process import RotationAndSegmentationTransform, HexaboardDataset
from autoencoder.model import SimpleCNNAutoEncoder
from autoencoder.training import evaluate_autoencoder

# Ensure deterministic behavior for GPU operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set dataset and checkpoint paths
DATASET_PATH = os.path.join(project_dir, 'datasets')
CHECKPOINT_PATH = os.path.join(autoencoder_dir, 'small_ae.pt')

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Run code
if __name__ == "__main__":
    # Read in the image
    image = Image.open(os.path.join(DATASET_PATH, 'unperturbed_images', 'hexaboard_1.png'))

    # Get the height and width of the image
    width, height = image.size

    # Adjust the number of segments
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # Read in and process the images
    train_dataset = HexaboardDataset(
        image_dir=os.path.join(DATASET_PATH, 'unperturbed_images'),
        transform=transform
    )
    val_dataset = HexaboardDataset(
        image_dir=os.path.join(DATASET_PATH, 'perturbed_segments'),
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Set the batch
    batch_size = 1

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Get the segments' height and width
    segment_height = train_dataset[0][0][0].shape[0]
    segment_width = train_dataset[0][0][0].shape[1]

    # Initialize the model
    cnn_ae = SimpleCNNAutoEncoder(
        height=segment_height,
        width=segment_width,
        latent_dim=128,
        kernel_sizes=[64, 128]
    )
    cnn_ae.to(device)

    # Load the model's weights
    cnn_ae.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))

    # Evaluate the model on unperturbed vs. perturbed images
    criterion = nn.BCEWithLogitsLoss()

    evaluate_autoencoder(
        model=cnn_ae,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=val_loader,
        num_images=8,
        visualize=True
    )