from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from .figure import finish_figure


def visualize_segment(
    data: np.ndarray,
    H_seg: int,
    V_seg: int,
    *,
    savefig: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[Figure]:
    """
    Display one RGB uint8 segment from a board.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    H_seg : int
        Input H seg.
    V_seg : int
        Input V seg.
    savefig : str or Path, optional
        Optional PNG path; create parent directories and save at 300 DPI.
    show : bool
        Display and close the figure when True; otherwise return it to the caller.

    Returns
    -------
    Figure or None
        Figure when `show=False`; otherwise None after display.
    """
    seg = data[H_seg, V_seg]  # (height, width, num_channels)
    img = Image.fromarray(seg.astype('uint8'))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    return finish_figure(plt.gcf(), savefig, show)


def plot_history(
    history: Dict[str, List[float]],
    save_fig: Optional[str] = None,
    *,
    savefig: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[Figure]:
    """
    Plot training and validation loss and metric history.

    Parameters
    ----------
    history : Dict[str, List[float]]
        Input history.
    save_fig : str, optional
        Compatibility alias for `savefig`.
    savefig : str or Path, optional
        Optional PNG path; create parent directories and save at 300 DPI.
    show : bool
        Display and close the figure when True; otherwise return it to the caller.

    Returns
    -------
    Figure or None
        Figure when `show=False`; otherwise None after display.
    """
    plt.figure(figsize=(12, 5))
    epochs = history['epoch']

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot training and validation metric (accuracy, for example)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_metric'], label="Train Accuracy")
    plt.plot(epochs, history['val_metric'], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    return finish_figure(plt.gcf(), savefig, show, save_fig)
