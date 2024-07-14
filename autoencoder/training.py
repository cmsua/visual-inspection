# Import necessary libraries
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from grokfast import gradfilter_ema

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

def train_autoencoder(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 100,
    chunk_size: int = 12,
    save_path: str = os.path.join(DATASET_PATH, 'model.pt')
) -> Tuple[Dict[str, List[float]], nn.Module]:
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }  # initialize a dictionary to store epoch-wise results
    best_loss_spread = 0.0

    # Initialize grads to None for the first iteration
    grads = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs = inputs.to(device)
            inputs = inputs.squeeze(0)
            for input in inputs.chunk(chunk_size):
                intput = input.squeeze(0)
                optimizer.zero_grad()

                output = model(input)
                loss = criterion(output, input)
                loss.backward()
                grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                optimizer.step()

                train_loss += loss.item()

        train_loss /= chunk_size * len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                inputs = inputs.squeeze(0)
                for input in inputs.chunk(chunk_size):
                    intput = input.squeeze(0)
                    output = model(input)
                    loss = criterion(output, input)

                    val_loss += loss.item()

            val_loss /= chunk_size * len(val_loader)

        # Append epoch results to history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(scheduler.get_last_lr())

        # Print results
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr() if scheduler else 'N/A'}"
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Save the parameters with the best validation loss
        if val_loss - train_loss > best_loss_spread:
            best_loss_spread = val_loss - train_loss
            torch.save(model.state_dict(), save_path)

    return history, model

def plot_metrics(history):
    epochs = history['epoch']

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_autoencoder(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: DataLoader,
    num_images: int,
    chunk_size: int,
    visualize: bool = False
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

    if visualize:
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

        plt.show()