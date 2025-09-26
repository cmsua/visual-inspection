from typing import Set, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_threshold_comparison(
    pw_metrics: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ae_metrics: Tuple[np.ndarray, np.ndarray, np.ndarray],
    skipped_segments: Optional[Set[Tuple[int, int]]] = None,
    save_fig: Optional[str] = None
) -> None:
    # Extract and validate inputs
    pw_opt, pw_bad, pw_good = pw_metrics
    ae_opt, ae_bad, ae_good = ae_metrics

    # Create a 2x4 grid where the last column is a single merged colorbar axis (spanning both rows).
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[1, 1, 1, 0.06],
        height_ratios=[1, 1],
        wspace=0.3,
        hspace=0.3
    )
    fig.subplots_adjust(left=0.04, right=0.96)

    # Last column is a single colorbar axis spanning both rows
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    cbar_ax = fig.add_subplot(gs[:, 3])

    # Row titles
    row_titles = [
        ["PW Good SSIMs", "PW Bad SSIMs", "PW Optimal Threshold"],
        ["AE Good SSIMs", "AE Bad SSIMs", "AE Optimal Threshold"]
    ]
    arrays_row0 = [pw_good, pw_bad, pw_opt]
    arrays_row1 = [ae_good, ae_bad, ae_opt]
    arrays = [arrays_row0, arrays_row1]

    # Default skip set
    if skipped_segments is None:
        skipped_segments = {
            (0, 0), (0, 1), (0, 7), (0, 8),
            (1, 0), (1, 8),
            (2, 0), (2, 8),
            (3, 0), (3, 8),
            (4, 0), (4, 8),
            (8, 0), (8, 8),
            (9, 0), (9, 8),
            (10, 0), (10, 1), (10, 7), (10, 8),
            (11, 0), (11, 1), (11, 8),
            (12, 0), (12, 1), (12, 7), (12, 8)
        }

    for r in range(2):
        for c in range(3):
            ax = axes[r][c]
            arr = arrays[r][c]
            t = row_titles[r][c]

            # Build skip mask (bounds-checked)
            H, W = arr.shape
            skip_mask = np.zeros((H, W), dtype=bool)
            for (h, v) in skipped_segments:
                if 0 <= h < H and 0 <= v < W:
                    skip_mask[h, v] = True

            # Prepare array with NaN at skipped positions so cmap's bad color can be used
            plot_arr = arr.astype(float).copy()
            plot_arr[skip_mask] = np.nan

            # Skip annotations for masked cells
            annot = np.full(arr.shape, '', dtype=object)
            for i in range(H):
                for j in range(W):
                    if not skip_mask[i, j] and np.isfinite(plot_arr[i, j]):
                        annot[i, j] = f"{plot_arr[i, j]:.3f}"

            # Use a copy of the colormap and set the bad color to black
            cmap = plt.cm.get_cmap('coolwarm')
            try:
                cmap = cmap.copy()
            except Exception:
                cmap = plt.cm.get_cmap('coolwarm')

            cmap.set_bad('black')

            # Determine whether to draw the colorbar only on the bottom-right plot (r==1, c==2)
            draw_cbar = (r == 1 and c == 2)
            sns.heatmap(
                data=plot_arr,
                vmin=0.0,
                vmax=1.0,
                cmap=cmap,
                annot=annot,
                fmt='',
                annot_kws={'fontsize': 8},
                cbar=draw_cbar,
                cbar_ax=(cbar_ax if draw_cbar else None),
                square=False,
                ax=ax
            )
            ax.set_title(t)
            ax.set_xlabel("V_seg")
            ax.set_ylabel("H_seg")

            # Set tick labels starting from 1 instead of 0
            ax.set_xticklabels([str(x) for x in np.arange(1, W + 1)])
            ax.set_yticklabels([str(y) for y in np.arange(1, H + 1)], rotation=0)

    fig.suptitle("Threshold Comparison")
    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()


def plot_confusion_matrices(
    ae_cm: np.ndarray,
    pw_cm: np.ndarray,
    double_cm: np.ndarray,
    title_prefix: str = '',
    save_fig: Optional[str] = None
) -> None:
    mats = [ae_cm, pw_cm, double_cm]
    names = ["Autoencoder", "Pixel-wise", "Both"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vmin = 0
    vmax = max(int(m.max()) for m in mats)
    for ax, mat, name in zip(axes, mats, names):
        # Convert to int for display
        display_mat = np.array(mat, dtype=int)
        sns.heatmap(
            display_mat,
            vmin=vmin,
            vmax=vmax,
            cmap='coolwarm',
            annot=True,
            fmt='d',
            linewidths=0.5,
            linecolor='gray',
            cbar=False,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Good", "Bad"])
        ax.set_yticklabels(["Good", "Bad"], rotation=0)
        ax.set_title(name)

    if title_prefix:
        fig.suptitle(title_prefix)

    fig.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()