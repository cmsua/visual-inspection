from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .model import load_inspection_model
from ..configs.inspection_config import InspectionConfig
from ..utils.data import HexaboardDataset
from ..utils.viz import plot_ae_comparison


def evaluate_reconstruction(config: InspectionConfig, segment_index: int = 0) -> Dict[str, float]:
    """
    Evaluate mean reconstruction loss without retaining the full dataset in RAM.

    Parameters
    ----------
    config : InspectionConfig
        Shared model and data settings.
    segment_index : int
        Retained segment index to illustrate from each dataset.

    Returns
    -------
    Dict[str, float]
        Normal and anomalous reconstruction losses.
    """
    config.resolve()
    datasets = {
        'normal': HexaboardDataset(config.test_dir, config.skipped_segments_path),
        'anomalous': HexaboardDataset(config.board_path, config.skipped_segments_path)
    }
    if any(not 0 <= segment_index < len(dataset) for dataset in datasets.values()):
        raise ValueError("Display segment index is outside one of the datasets.")

    device = torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    sample = datasets['normal']
    model = load_inspection_model(config.checkpoint_path, sample.height, sample.width, device, config.model)
    losses = {}
    examples = {}
    with torch.inference_mode():
        for name, dataset in datasets.items():
            total = 0.0
            for index, batch in enumerate(DataLoader(dataset, batch_size=1)):
                batch = batch.to(device)
                outputs = model(batch)
                total += functional.binary_cross_entropy_with_logits(outputs, batch).item()
                if index == segment_index:
                    examples[name] = (batch.cpu().numpy(), torch.sigmoid(outputs).cpu().numpy())

            losses[name] = total / len(dataset)

    figure = plot_ae_comparison(
        *examples['normal'],
        *examples['anomalous'],
        segment_idx=0,
        savefig=str(Path(config.output_dir) / 'reconstruction.png'),
        show=False
    )
    plt.close(figure)
    return losses
