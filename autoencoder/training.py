# Import necessary libraries
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Set the seed
torch.manual_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to the checkpoints folder
# CHECKPOINT_PATH = '/content/drive/MyDrive/Colab Notebooks/HGCAL/checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_autoencoder(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 100,
    chunk_size: int = 12,
    save_path: str = 'model.pt',
    device: torch.device = torch.device('cpu')
) -> Tuple[Dict[str, List[float]], nn.Module]:
    """
    Trains an autoencoder model and evaluates its performance on a validation set.

    Args:
        model (nn.Module): Autoencoder model to train.
        criterion (nn.Module): Loss function to minimize.
        optimizer (optim.Optimizer): Optimizer for the training process.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        scheduler (Optional[optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler. Defaults to None.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 100.
        chunk_size (int, optional): Number of chunks to split inputs for mini-batching. Defaults to 12.
        save_path (str, optional): Path to save the best model. Defaults to 'model.pt'.
        device (torch.device, optional): Device on which to perform training. Defaults to CPU.

    Returns:
        (history, model) (Tuple[Dict[str, List[float]], nn.Module]): Training history and the trained model.
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs in train_loader:
            inputs = inputs.to(device)
            inputs = inputs.squeeze(0)

            for input in inputs.chunk(chunk_size):
                input = input.squeeze(0)
                optimizer.zero_grad()

                # Forward pass
                output = model(input)
                loss = criterion(output, input)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        train_loss /= chunk_size * len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)

                # Forward pass
                output = model(inputs)
                loss = criterion(output, inputs)
                
                val_loss += loss.item()

            val_loss /= len(val_loader)

        # Append epoch results to history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Get the current learning rate
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)

        # Print results
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    return history, model

def plot_metrics(history: Dict[str, List[float]]) -> None:
    """
    Plots the training and validation loss along with the learning rate schedule.

    Args:
        history (Dict[str, List[float]]): A dictionary containing the epoch-wise metrics.
            Keys should include 'epoch', 'train_loss', 'val_loss', and 'learning_rate'.
    """
    epochs = history['epoch']

    # Create the figure
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot learning rate schedule
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['learning_rate'], label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    # Adjust layout to avoid overlap and display the plots
    plt.tight_layout()
    plt.show()

def evaluate_autoencoder(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_images: int,
    device: torch.device = torch.device('cpu'),
    visualize: bool = False
) -> None:
    """
    Evaluates the autoencoder on training and testing data, calculating loss and optionally visualizing the original
    and reconstructed images.

    Args:
        model (nn.Module): The trained autoencoder model to evaluate.
        criterion (nn.Module): The loss function to evaluate reconstruction error.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        num_images (int): Number of images to visualize for original vs. reconstructed comparison.
        device (torch.device, optional): The device to use for evaluation (CPU or GPU). Defaults to CPU.
        visualize (bool, optional): Whether to display the original and reconstructed images. Defaults to False.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0

    original_images = []
    reconstructed_images = []

    with torch.no_grad():
        # Evaluate on training data
        for inp in train_loader:
            inp = inp.squeeze(0).to(device)
            for inputs in inp:
                inputs = inputs.unsqueeze(0)  # Add batch dimension
                output = model(inputs)
                loss = criterion(output, inputs)
                total_loss += loss.item()

                # Save some images for visualization
                if len(original_images) < num_images:
                    original_images.append(inputs.cpu())
                    reconstructed_images.append(torch.sigmoid(output.cpu()))

        train_loss = total_loss / inp.size(0)  # Average loss per image
        print(f'Train Loss: {train_loss:.4f}')

    # Visualize the images
    if visualize:
        visualize_images(original_images, reconstructed_images, num_images, title='Training')

    # Reset for test set evaluation
    total_loss = 0.0
    original_images = []
    reconstructed_images = []

    with torch.no_grad():
        # Evaluate on testing data
        for inputs in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            loss = criterion(output, inputs)
            total_loss += loss.item()

            # Save some images for visualization
            if len(original_images) < num_images:
                original_images.append(inputs.cpu())
                reconstructed_images.append(torch.sigmoid(output.cpu()))

        test_loss = total_loss / len(test_loader)  # Average loss over the entire test set
        print(f'Test Loss: {test_loss:.4f}')

    # Visualize the images
    if visualize:
        visualize_images(original_images, reconstructed_images, num_images, title='Testing')


def visualize_images(
    original_images: List[torch.Tensor],
    reconstructed_images: List[torch.Tensor],
    num_images: int,
    title: str
) -> None:
    """
    Helper function to visualize original and reconstructed images.

    Args:
        original_images (List[torch.Tensor]): List of original images.
        reconstructed_images (List[torch.Tensor]): List of reconstructed images.
        num_images (int): Number of images to visualize.
        title (str): Title for the set of images (Training/Testing).
    """
    plt.figure(figsize=(15, 4))

    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        img = original_images[i][0].permute(1, 2, 0).numpy()
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        if i == 0:
            plt.title(f'Original ({title})')

        # Reconstructed images
        plt.subplot(2, num_images, i + 1 + num_images)
        img = reconstructed_images[i][0].permute(1, 2, 0).numpy()
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        if i == 0:
            plt.title(f'Reconstructed ({title})')

    plt.tight_layout()
    plt.show()