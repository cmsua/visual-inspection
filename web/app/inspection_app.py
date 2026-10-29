import os
import shlex
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import yaml

from src.configs import DEFAULT_INSPECTION_CONFIG, REPO_ROOT, InspectionConfig
from src.inspection import inspect_board, validate_calibration
from src.utils import InspectionResults
from src.utils.data import JSONStore, load_skipped_segments, open_board, rgb_segment, validate_coordinates

from ..utils import encode_png, repo_relative


class InspectionApp:
    def __init__(self, config: Optional[InspectionConfig] = None, labels_path: Optional[str] = None) -> None:
        self.config = (config or InspectionConfig.from_yaml()).resolve()
        self._lock = threading.RLock()
        self._execution_lock = threading.Lock()
        self._json_store = JSONStore(labels_path or self.config.labels_path)

    @contextmanager
    def execution_guard(self) -> Generator[None, None, None]:
        if not self._execution_lock.acquire(blocking=False):
            raise ValueError("An inspection or agent job is running. Wait or stop it first.")

        try:
            yield
        finally:
            self._execution_lock.release()

    def resolve_board_path(self, value: Optional[str] = None) -> Path:
        candidate = (REPO_ROOT / (value or self.config.board_path)).resolve()
        root = Path(self.config.data_root).resolve()
        if not candidate.is_relative_to(root) or candidate.suffix.lower() != '.npy':
            raise ValueError("Board must be a `.npy` file inside the configured data directory.")

        if not candidate.is_file():
            matches = sorted(
                path.resolve() for path in root.rglob('*.npy')
                if path.name == candidate.name and path.is_file() and path.resolve().is_relative_to(root)
            )
            message = f"Board file not found: `{candidate}`. Check the folder as well as the board filename."
            if matches:
                alternatives = ', '.join(f'`{repo_relative(path)}`' for path in matches[:5])
                message += f" Matching filenames exist at: {alternatives}."

            raise FileNotFoundError(message)

        return candidate

    def list_boards(self, folder: Optional[str] = None) -> Dict[str, object]:
        root = Path(self.config.data_root).resolve()
        default = root / 'bad' if (root / 'bad').is_dir() else root
        directory = (REPO_ROOT / folder).resolve() if folder else default.resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Board folder must be inside the configured data directory.")
        if not directory.is_dir():
            raise FileNotFoundError(f"Board folder not found: `{directory}`.")

        boards = []
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() == '.npy' and path.is_file() and path.resolve().is_relative_to(root):
                boards.append({
                    'name': path.relative_to(directory).as_posix(),
                    'path': repo_relative(path.resolve())
                })

        return {'folder': repo_relative(directory), 'boards': boards}

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

    def build_board_payload(
        self,
        board_path: Path,
        inspection: Optional[InspectionResults] = None,
        run_inspection: bool = True
    ) -> Dict[str, object]:
        array = self._load_board(board_path)
        rows, cols, height, width, channels = array.shape
        if inspection is None and run_inspection:
            inspection = self.get_inspection(board_path)

        skipped = load_skipped_segments(self.config.skipped_segments_path)
        unknown = [[-1 if (row, col) in skipped else None for col in range(cols)] for row in range(rows)]
        scores = inspection.to_dict() if inspection is not None else {
            'pw_flags': unknown,
            'ae_flags': unknown,
            'hybrid_flags': unknown,
            'metadata': {'inspection_status': 'not_run'}
        }
        return {
            'board_path': repo_relative(board_path),
            'baseline_path': repo_relative(Path(self.config.baseline_path)),
            'grid_shape': {'rows': rows, 'cols': cols},
            'segment_shape': {'height': height, 'width': width, 'channels': channels},
            'damaged_segments': self.get_damaged_labels(board_path, array.shape),
            'skipped_segments': [{'row': row, 'col': col} for row, col in sorted(skipped)],
            'inspection': scores
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
        try:
            validate_calibration(self.config)
        except (ValueError, OSError) as error:
            command = shlex.join([
                sys.executable, '-m', 'scripts.calibrate',
                '--config-path', self.config.source_path or str(DEFAULT_INSPECTION_CONFIG)
            ])
            raise ValueError(
                f"Inspection is blocked by incompatible or missing calibration: {error} "
                f"Regenerate thresholds from the repository root with: {command}. "
                "Then retry Run inspection. Manual review remains available."
            ) from error

        with self._lock, tempfile.TemporaryDirectory(prefix='hexaboard_job_') as directory:
            # Serialize exactly the settings used by this running server.
            snapshot = Path(directory) / 'inspection.yaml'
            settings = asdict(self.config)
            settings.pop('source_path')
            snapshot.write_text(yaml.safe_dump({'inspection': settings}), encoding='utf-8')
            if os.name == 'nt':
                command = [
                    sys.executable,
                    '-m',
                    'scripts.inspect',
                    '--config-path',
                    str(snapshot),
                    '-n',
                    str(board_path),
                    '--force'
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
