import random
from PIL import Image
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def visualize_segments(
    data: np.ndarray,
    num_samples: int = 4,
    randomize: bool = True
) -> None:
    """
    Visualize a few segments from a 5D hexaboard array using PIL.

    Parameters
    ----------
    data : np.ndarray
        shape (H_seg, V_seg, height, width, num_channels), e.g. (12, 9, 1016, 1640, 3)
    num_samples : int
        how many segments to display
    randomize : bool
        whether to pick segments at random or use fixed corners
    """

    H_seg, V_seg, _, _, _ = data.shape

    # Pick indices to show
    if randomize:
        indices = [
            (random.randrange(H_seg), random.randrange(V_seg))
            for _ in range(num_samples)
        ]
    else:
        corners = [
            (0, 0),
            (0, V_seg - 1),
            (H_seg - 1, 0),
            (H_seg - 1, V_seg - 1),
        ]
        indices = corners[:num_samples]

    for _, (h_idx, v_idx) in enumerate(indices, 1):
        seg = data[h_idx, v_idx]  # (height, width, num_channels)
        img = Image.fromarray(seg.astype('uint8'))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def plot_reconstructions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_images: int = 8,
    title: str = "Testing Dataset"
) -> None:
    """
    Visualize original vs. reconstructed images from numpy arrays,
    arranged as two columns: originals on the left, reconstructions on the right.

    Parameters
    ----------
    y_true : np.ndarray
        Array of shape (N, C, H, W) containing original images.
    y_pred : np.ndarray
        Array of shape (N, C, H, W) containing reconstructed images.
    num_images : int, optional
        Number of image pairs to display (default: 8).
    title : str, optional
        Title for the visualization (default: "Testing Dataset").
    """
    # Clip num_images to available samples
    n = min(num_images, y_true.shape[0])
    # one row per pair, two columns
    plt.figure(figsize=(6, 3 * n))

    for i in range(n):
        # Original in left column
        plt.subplot(n, 2, 2 * i + 1)
        img_orig = np.transpose(y_true[i], (1, 2, 0))  # C, H, W -> H, W, C
        plt.imshow(np.clip(img_orig, 0, 1))
        plt.axis('off')
        if i == 0:
            plt.title(f'Original ({title})')

        # Reconstructed in right column
        plt.subplot(n, 2, 2 * i + 2)
        img_recon = np.transpose(y_pred[i], (1, 2, 0))
        plt.imshow(np.clip(img_recon, 0, 1))
        plt.axis('off')
        if i == 0:
            plt.title(f'Reconstructed ({title})')

    plt.tight_layout(h_pad=0.25, w_pad=0.25)
    plt.show()


def plot_history(history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 5))
    epochs = history['epoch']

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation metric (accuracy, for example)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_metric'], label="Train Accuracy")
    plt.plot(epochs, history['val_metric'], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()