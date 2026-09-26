from typing import Iterable, Iterator, Tuple

import numpy as np
from skimage.metrics import structural_similarity

import torch
import torch.nn.functional as functional
from torch import nn

from .data import rgb_segment


def segment_scores(
    baseline: np.ndarray,
    board: np.ndarray,
    coordinates: Iterable[Tuple[int, int]],
    model: nn.Module,
    device: torch.device
) -> Iterator[Tuple[int, int, float, float]]:
    # Keep only the current segment and reconstruction in memory.
    for row, col in coordinates:
        original = rgb_segment(board, row, col)
        reference = rgb_segment(baseline, row, col)
        pixel_score = 1 - structural_similarity(reference, original, data_range=1.0, channel_axis=2)
        with torch.inference_mode():
            tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).to(device)
            prediction = torch.sigmoid(model(tensor))
            if prediction.shape != tensor.shape or not torch.isfinite(prediction).all():
                raise ValueError("Model returned non-finite pixels or an incompatible reconstruction shape.")

            ae_score = functional.l1_loss(prediction, tensor).item()

        yield row, col, float(pixel_score), float(ae_score)
