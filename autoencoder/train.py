# Import necessary dependencies
import os

from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Define project directory
project_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
autoencoder_dir = os.path.join(project_dir, 'autoencoder')

# Import modules using absolute imports
from preprocessing.ae_process import RotationAndSegmentationTransform, HexaboardDataset
from autoencoder.model import SimpleCNNAutoEncoder
from autoencoder.training import train_autoencoder, plot_metrics

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
    # print('Image width:', width)
    # print('Image height:', height)

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
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # # Print some information about the data
    # print(f'Train dataset size: {len(train_dataset)}')
    # print(f'Validation dataset size: {len(val_dataset)}')
    # print(f'Test dataset size: {len(test_dataset)}')
    # print(f'Segments Shape: {val_dataset[0].shape}')
    # print(f'Image shape: {val_dataset[0][0].shape}')
    # print(f'Image tensor type: {train_dataset[0][0].dtype}')
    # print(f'Batches: {len(train_loader)}')

    # Get the segments' height and width
    segment_height = train_dataset[0][0][0].shape[0]
    segment_width = train_dataset[0][0][0].shape[1]
    # print('Segment height:', segment_height)
    # print('Segment width:', segment_width)

    # Initialize the model
    cnn_ae = SimpleCNNAutoEncoder(
        height=segment_height,
        width=segment_width,
        latent_dim=128,
        kernel_sizes=[64, 128]
    )
    cnn_ae.to(device)

    # Model parameters
    optimizer = optim.Adam(cnn_ae.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    history, cnn_ae = train_autoencoder(
        model=cnn_ae,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        num_epochs=500,
        save_path=CHECKPOINT_PATH
    )

    # View the training progress
    plot_metrics(history)