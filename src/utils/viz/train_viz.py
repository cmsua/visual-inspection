from PIL import Image
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def visualize_segment(
    data: np.ndarray,
    H_seg: int,
    V_seg: int
) -> None:
    seg = data[H_seg, V_seg]  # (height, width, num_channels)
    img = Image.fromarray(seg.astype('uint8'))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_history(history: Dict[str, List[float]], save_fig: Optional[str] = None) -> None:
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
    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()