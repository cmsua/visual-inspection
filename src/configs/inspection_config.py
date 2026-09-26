from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSPECTION_CONFIG = REPO_ROOT / 'configs' / 'inspection.yaml'


@dataclass
class InspectionConfig:
    """
    Shared configuration for calibration, inspection, analysis, and HTTP serving.

    Paths in YAML are relative to the repository root. Model parameters must
    match the saved weights; omitted parameters are inferred from CNN weights.

    Parameters
    ----------
    baseline_path : str
        Reference board used for pixel comparisons.
    checkpoint_path : str
        Best-model weights or a resumable trainer checkpoint.
    calibration_dir : str
        Threshold arrays and calibration manifest directory.
    """

    baseline_path: str = 'data/train/320XLF4CQH00001.npy'
    board_path: str = 'data/bad_example/320XLF4CQH00445.npy'
    data_root: str = 'data'
    checkpoint_path: str = 'logs/CNNAutoencoder/best/run_01.pt'
    skipped_segments_path: str = 'calibrations/skipped_segments.json'
    labels_path: str = 'calibrations/damaged_segments.json'
    calibration_dir: str = 'calibrations'
    output_dir: str = 'outputs/inspection'
    good_dirs: List[str] = field(default_factory=lambda: ['data/train', 'data/val'])
    bad_dir: str = 'data/bad'
    test_dir: str = 'data/test'
    model: Dict = field(default_factory=dict)
    device: Optional[str] = None
    bad_fraction: float = 0.5
    normal_quantile: float = 0.99
    seed: int = 42
    source_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str = str(DEFAULT_INSPECTION_CONFIG)) -> 'InspectionConfig':
        """
        Read reusable inspection settings.

        Parameters
        ----------
        path : str, optional
            YAML configuration path.

        Returns
        -------
        InspectionConfig
            Resolved settings with absolute paths.
        """
        with open(path, 'r', encoding='utf-8') as handle:
            document = yaml.safe_load(handle)

        config = cls(**document['inspection'])
        config.source_path = str(Path(path).resolve())
        return config.resolve()

    def resolve(self) -> 'InspectionConfig':
        for name in (
            'baseline_path',
            'board_path',
            'data_root',
            'checkpoint_path',
            'skipped_segments_path',
            'labels_path',
            'calibration_dir',
            'output_dir',
            'bad_dir',
            'test_dir'
        ):
            setattr(self, name, str((REPO_ROOT / getattr(self, name)).resolve()))

        self.good_dirs = [str((REPO_ROOT / path).resolve()) for path in self.good_dirs]
        if not 0 < self.bad_fraction <= 1:
            raise ValueError("`bad_fraction` must be in `(0, 1]`.")

        if not 0 < self.normal_quantile < 1:
            raise ValueError("`normal_quantile` must be in `(0, 1)`.")

        return self
