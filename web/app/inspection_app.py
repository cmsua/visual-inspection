import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.configs.inspection_config import REPO_ROOT, InspectionConfig
from src.inspection import inspect_board
from src.utils.data import JSONStore, load_skipped_segments
from src.utils.data.board_contract import open_board, rgb_segment, validate_coordinates
from src.utils.get_results import InspectionResults

from ..utils import encode_png, repo_relative


class InspectionApp:
    def __init__(self, config: Optional[InspectionConfig] = None) -> None:
        self.config = (config or InspectionConfig.from_yaml()).resolve()
        self._lock = threading.RLock()
        self._json_store = JSONStore(self.config.labels_path)

    def resolve_board_path(self, value: Optional[str] = None) -> Path:
        candidate = (REPO_ROOT / (value or self.config.board_path)).resolve()
        root = Path(self.config.data_root).resolve()
        if not candidate.is_relative_to(root) or candidate.suffix.lower() != '.npy':
            raise ValueError("Board must be a `.npy` file inside the configured data directory.")

        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        return candidate

    def _load_board(self, board_path: Path) -> np.ndarray:
        return open_board(str(self.resolve_board_path(str(board_path))))

    def get_inspection(self, board_path: Path, force: bool = False) -> InspectionResults:
        with self._lock:
            return inspect_board(self.config, str(board_path), force=force)

    def get_damaged_labels(self, board_path: Path, shape: Tuple[int, ...]) -> List[Dict[str, int]]:
        with self._lock:
            self._json_store._load()
            damaged = self._json_store.get_damaged_set(repo_relative(board_path))
            validate_coordinates(damaged, shape[:2])
            return [{'row': r, 'col': c} for r, c in sorted(damaged)]

    def update_damaged_label(
        self,
        board_path: Path,
        shape: Tuple[int, ...],
        row: int,
        col: int,
        is_damaged: bool
    ) -> None:
        validate_coordinates([(row, col)], shape[:2])
        with self._lock:
            self._json_store._load()
            board_id = repo_relative(board_path)
            self._json_store.ensure_file_entry(board_id, shape)
            damaged = self._json_store.get_damaged_set(board_id)
            if is_damaged:
                damaged.add((row, col))
            else:
                damaged.discard((row, col))

            self._json_store.save_damaged_set(board_id, damaged)
            self._json_store.save_checkpoint(board_id, row=row, col=col, idx=row * shape[1] + col)

    def build_board_payload(self, board_path: Path) -> Dict[str, object]:
        array = self._load_board(board_path)
        rows, cols, height, width, channels = array.shape
        inspection = self.get_inspection(board_path)
        skipped = load_skipped_segments(self.config.skipped_segments_path)
        return {
            'board_path': repo_relative(board_path),
            'baseline_path': repo_relative(Path(self.config.baseline_path)),
            'grid_shape': {'rows': rows, 'cols': cols},
            'segment_shape': {'height': height, 'width': width, 'channels': channels},
            'damaged_segments': self.get_damaged_labels(board_path, array.shape),
            'skipped_segments': [{'row': row, 'col': col} for row, col in sorted(skipped)],
            'inspection': inspection.to_dict()
        }

    def segment_image_payload(self, board_path: Path, row: int, col: int) -> Dict[str, object]:
        array = self._load_board(board_path)
        validate_coordinates([(row, col)], array.shape[:2])
        return {
            'board_path': repo_relative(board_path),
            'row': row,
            'col': col,
            'image': encode_png(rgb_segment(array, row, col, normalize=False))
        }

    def run_inspection_job(self, board_path: Path) -> subprocess.CompletedProcess:
        with self._lock, tempfile.TemporaryDirectory(prefix='hexaboard_job_') as directory:
            # Serialize exactly the settings used by this running server.
            snapshot = Path(directory) / 'inspection.yaml'
            settings = asdict(self.config)
            settings.pop('source_path')
            snapshot.write_text(yaml.safe_dump({'inspection': settings}), encoding='utf-8')
            if os.name == 'nt':
                command = [
                    sys.executable, '-m', 'scripts.inspect', '--config-path',
                    str(snapshot), '-n', str(board_path), '--force'
                ]
            else:
                command = ['bash', str(REPO_ROOT / 'jobs' / 'main.sh'), str(board_path)]

            environment = dict(os.environ)
            environment['INSPECTION_PYTHON'] = sys.executable
            environment['INSPECTION_CONFIG'] = str(snapshot)
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(REPO_ROOT),
                env=environment
            )
