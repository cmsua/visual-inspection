from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from skimage.metrics import structural_similarity as ssim

from .figure import finish_figure


def plot_reconstructions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_images: int = 8,
    title: str = "Testing Dataset",
    *,
    savefig: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[Figure]:
    """
    Visualize original vs. reconstructed images from numpy arrays,
    arranged as two columns: originals on the left, reconstructions on the right.

    Parameters
    ----------
    y_true : np.ndarray
        Array of shape `[N, C, H, W]` containing original images.
    y_pred : np.ndarray
        Array of shape `[N, C, H, W]` containing reconstructed images.
    num_images : int, optional
        Number of image pairs to display (default: 8).
    title : str, optional
        Title for the visualization (default: "Testing Dataset").
    """
    # Clip num_images to available samples
    n = min(num_images, y_true.shape[0])

    # One row per pair, two columns
    plt.figure(figsize=(6, 3 * n))

    for i in range(n):
        # Original in left column
        plt.subplot(n, 2, 2 * i + 1)
        img_orig = np.transpose(y_true[i], (1, 2, 0))  # C, H, W -> H, W, C
        plt.imshow(np.clip(img_orig, 0, 1))
        plt.axis('off')

        if i == 0:
            plt.title(f"Original ({title})")

        # Reconstructed in right column
        plt.subplot(n, 2, 2 * i + 2)
        img_recon = np.transpose(y_pred[i], (1, 2, 0))
        plt.imshow(np.clip(img_recon, 0, 1))
        plt.axis('off')

        if i == 0:
            plt.title(f"Reconstructed ({title})")

    plt.tight_layout(h_pad=0.25, w_pad=0.25)
    return finish_figure(plt.gcf(), savefig, show)


def plot_pw_comparison(
    reference_image: np.ndarray,
    good_image: np.ndarray,
    bad_image: np.ndarray,
    H_seg: int,
    V_seg: int,
    save_fig: Optional[str] = None,
    *,
    savefig: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[Figure]:
    """
    Compare raw BGR board segments using SSIM.

    Parameters
    ----------
    reference_image : np.ndarray
        Input reference image.
    good_image : np.ndarray
        Input good image.
    bad_image : np.ndarray
        Input bad image.
    H_seg : int
        Input H seg.
    V_seg : int
        Input V seg.
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
    reference_image = reference_image[..., ::-1].copy()
    good_image = good_image[..., ::-1].copy()
    bad_image = bad_image[..., ::-1].copy()

    reference_segment = reference_image[H_seg, V_seg] / 255.0
    good_segment = good_image[H_seg, V_seg] / 255.0
    bad_segment = bad_image[H_seg, V_seg] / 255.0

    # Compute structural similarity index (SSIM)
    m1, diff1 = ssim(reference_segment, good_segment, data_range=1.0, channel_axis=2, full=True)
    m2, diff2 = ssim(reference_segment, bad_segment, data_range=1.0, channel_axis=2, full=True)

    # Scale SSIM difference to [0, 1]
    diff1 = np.abs(diff1)
    diff2 = np.abs(diff2)

    _, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=300)

    # Reference segment
    axes[0, 0].imshow(reference_segment)
    axes[0, 0].set_title("Reference Segment")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(reference_segment)
    axes[1, 0].set_title("Reference Segment")
    axes[1, 0].axis('off')

    # Good & bad segments
    axes[0, 1].imshow(good_segment)
    axes[0, 1].set_title("Good Segment")
    axes[0, 1].axis('off')

    axes[1, 1].imshow(bad_segment)
    axes[1, 1].set_title("Bad Segment")
    axes[1, 1].axis('off')

    # SSIM Differences
    axes[0, 2].imshow(np.clip(diff1, 0, 1), vmin=0, vmax=1)
    axes[0, 2].set_title("SSIM Difference")
    axes[0, 2].axis('off')
    axes[0, 2].text(
        0.05,
        0.95,
        f"SSIM: {m1:.3f}",
        transform=axes[0, 2].transAxes,
        color='black',
        fontsize=12,
        va='top',
        bbox=dict(facecolor='white', alpha=1.0, pad=4)
    )

    axes[1, 2].imshow(np.clip(diff2, 0, 1), vmin=0, vmax=1)
    axes[1, 2].set_title("SSIM Difference")
    axes[1, 2].axis('off')
    axes[1, 2].text(
        0.05,
        0.95,
        f"SSIM: {m2:.3f}",
        transform=axes[1, 2].transAxes,
        color='black',
        fontsize=12,
        va='top',
        bbox=dict(facecolor='white', alpha=1.0, pad=4)
    )

    plt.tight_layout()
    return finish_figure(plt.gcf(), savefig, show, save_fig)


def plot_ae_comparison(
    y_true_good: np.ndarray,
    y_pred_good: np.ndarray,
    y_true_bad: np.ndarray,
    y_pred_bad: np.ndarray,
    segment_idx: int,
    save_fig: Optional[str] = None,
    *,
    savefig: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[Figure]:
    # Get the segment to visualize
    """
    Compare normal and anomalous reconstructions with MAE residuals.

    Parameters
    ----------
    y_true_good : np.ndarray
        Input y true good.
    y_pred_good : np.ndarray
        Input y pred good.
    y_true_bad : np.ndarray
        Input y true bad.
    y_pred_bad : np.ndarray
        Input y pred bad.
    segment_idx : int
        Input segment idx.
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
    seg_good = y_true_good[segment_idx]
    recon_seg_good = y_pred_good[segment_idx]
    seg_bad = y_true_bad[segment_idx]
    recon_seg_bad = y_pred_bad[segment_idx]

    # Change from PyTorch's image dimension to Numpy's: (C, H, W) -> (H, W, C)
    seg_good = np.transpose(seg_good, (1, 2, 0))
    recon_seg_good = np.transpose(recon_seg_good, (1, 2, 0))
    seg_bad = np.transpose(seg_bad, (1, 2, 0))
    recon_seg_bad = np.transpose(recon_seg_bad, (1, 2, 0))

    # # Compute structural similarity index (SSIM)
    # m1, diff1 = ssim(seg_good, recon_seg_good, data_range=1.0, channel_axis=2, full=True)
    # m2, diff2 = ssim(seg_bad, recon_seg_bad, data_range=1.0, channel_axis=2, full=True)
    m1 = np.abs(seg_good - recon_seg_good).mean()
    m2 = np.abs(seg_bad - recon_seg_bad).mean()
    diff1 = np.abs(seg_good - recon_seg_good)
    diff2 = np.abs(seg_bad - recon_seg_bad)

    _, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=300)

    # Original segments
    axes[0, 0].imshow(np.clip(seg_good, 0, 1))
    axes[0, 0].set_title("Original")
    axes[0, 0].set_ylabel("Good Board")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(np.clip(seg_bad, 0, 1))
    axes[1, 0].set_ylabel("Anomalous Board")
    axes[1, 0].axis('off')

    # Reconstructed segments
    axes[0, 1].imshow(np.clip(recon_seg_good, 0, 1))
    axes[0, 1].set_title("Autoencoder Output")
    axes[0, 1].axis('off')

    axes[1, 1].imshow(np.clip(recon_seg_bad, 0, 1))
    axes[1, 1].axis('off')

    # Differences between the original and reconstructed images
    axes[0, 2].imshow(np.clip(diff1, 0, 1), vmin=0, vmax=1)
    axes[0, 2].set_title("Differences")
    axes[0, 2].axis('off')
    axes[0, 2].text(
        0.05,
        0.95,
        f"MAE: {m1:.3f}",
        transform=axes[0, 2].transAxes,
        color='black',
        fontsize=12,
        va='top',
        bbox=dict(facecolor='white', alpha=1.0, pad=4)
    )

    axes[1, 2].imshow(np.clip(diff2, 0, 1), vmin=0, vmax=1)
    axes[1, 2].axis('off')
    axes[1, 2].text(
        0.05,
        0.95,
        f"MAE: {m2:.3f}",
        transform=axes[1, 2].transAxes,
        color='black',
        fontsize=12,
        va='top',
        bbox=dict(facecolor='white', alpha=1.0, pad=4)
    )

    plt.tight_layout()
    return finish_figure(plt.gcf(), savefig, show, save_fig)
