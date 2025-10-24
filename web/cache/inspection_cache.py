from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ..utils import repo_relative
from src.utils import InspectionResults


@dataclass
class InspectionCache:
    board_path: Path
    baseline_path: Path
    inspection: InspectionResults
    board_mtime: float
    ae_threshold_mtime: float
    pw_threshold_mtime: float
    skipped_mtime: float

    def as_payload(self) -> Dict[str, object]:
        return {
            'pw_flags': self.inspection.pw_flags.tolist(),
            'ae_flags': self.inspection.ae_flags.tolist(),
            'hybrid_flags': self.inspection.hybrid_flags.tolist(),
            'metadata': self.inspection.metadata,
            'baseline_path': repo_relative(self.baseline_path),
            'inspected_path': repo_relative(self.board_path),
        }
