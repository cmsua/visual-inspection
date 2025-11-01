from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..cache import InspectionCache
from ..configs import (
    AE_THRESHOLD_PATH,
    BASELINE_BOARD_PATH,
    DAMAGED_SEGMENTS_PATH,
    PW_THRESHOLD_PATH,
    SKIPPED_SEGMENTS_PATH,
    REPO_ROOT
)
from ..utils import encode_png, repo_relative, to_wsl_path
from src.inspection import run_inspection
from src.utils.data import JSONStore, load_hexaboard, load_skipped_segments


class InspectionApp:
    """Stateful helper attached to the HTTP server."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._json_store = JSONStore(str(DAMAGED_SEGMENTS_PATH))
        self._board_cache = {}
        self._inspection_cache = {}

    def _load_board(self, board_path: Path) -> np.ndarray:
        with self._lock:
            cached = self._board_cache.get(board_path)
            mtime = board_path.stat().st_mtime
            if cached and cached['mtime'] >= mtime:
                return cached['array']

            array = load_hexaboard(str(board_path), normalize=False)
            self._board_cache[board_path] = {'array': array, 'mtime': mtime}
            return array

    @staticmethod
    def _board_identifier(board_path: Path) -> str:
        return repo_relative(board_path)

    def _needs_refresh(self, board_path: Path) -> bool:
        cache_entry = self._inspection_cache.get(board_path)
        if cache_entry is None:
            return True

        board_mtime = board_path.stat().st_mtime
        ae_mtime = AE_THRESHOLD_PATH.stat().st_mtime
        pw_mtime = PW_THRESHOLD_PATH.stat().st_mtime
        skipped_mtime = SKIPPED_SEGMENTS_PATH.stat().st_mtime

        if board_mtime > cache_entry.board_mtime:
            return True
        if ae_mtime > cache_entry.ae_threshold_mtime:
            return True
        if pw_mtime > cache_entry.pw_threshold_mtime:
            return True
        if skipped_mtime > cache_entry.skipped_mtime:
            return True

        return False

    def get_inspection(self, board_path: Path, force: bool = False) -> InspectionCache:
        with self._lock:
            if force or self._needs_refresh(board_path):
                inspection = run_inspection(
                    baseline_hexaboard_path=str(BASELINE_BOARD_PATH),
                    new_hexaboard_path=str(board_path),
                    skipped_segments_path=str(SKIPPED_SEGMENTS_PATH),
                    ae_threshold_path=str(AE_THRESHOLD_PATH),
                    pw_threshold_path=str(PW_THRESHOLD_PATH),
                )
                cache_entry = InspectionCache(
                    board_path=board_path,
                    baseline_path=BASELINE_BOARD_PATH,
                    inspection=inspection,
                    board_mtime=board_path.stat().st_mtime,
                    ae_threshold_mtime=AE_THRESHOLD_PATH.stat().st_mtime,
                    pw_threshold_mtime=PW_THRESHOLD_PATH.stat().st_mtime,
                    skipped_mtime=SKIPPED_SEGMENTS_PATH.stat().st_mtime,
                )
                self._inspection_cache[board_path] = cache_entry
                return cache_entry

            return self._inspection_cache[board_path]

    def get_damaged_labels(self, board_path: Path, shape: Tuple[int, ...]) -> List[Dict[str, int]]:
        board_id = self._board_identifier(board_path)
        self._json_store.ensure_file_entry(board_id, shape)
        damaged = self._json_store.get_damaged_set(board_id)
        return [{'row': int(r), 'col': int(c)} for (r, c) in sorted(damaged)]

    def update_damaged_label(self, board_path: Path, shape: Tuple[int, ...], row: int, col: int, is_damaged: bool) -> None:
        board_id = self._board_identifier(board_path)
        self._json_store.ensure_file_entry(board_id, shape)
        damaged = self._json_store.get_damaged_set(board_id)
        index = (int(row), int(col))

        if is_damaged:
            damaged.add(index)
        else:
            damaged.discard(index)

        self._json_store.save_damaged_set(board_id, damaged)
        self._json_store.save_checkpoint(board_id, row=int(row), col=int(col), idx=int(row) * shape[1] + int(col))

    def build_board_payload(self, board_path: Path) -> Dict[str, object]:
        array = self._load_board(board_path)
        H, W, height, width, channels = array.shape
        inspection = self.get_inspection(board_path)
        skipped_segments = load_skipped_segments(str(SKIPPED_SEGMENTS_PATH))

        payload = {
            'board_path': repo_relative(board_path),
            'baseline_path': repo_relative(BASELINE_BOARD_PATH),
            'grid_shape': {'rows': H, 'cols': W},
            'segment_shape': {'height': height, 'width': width, 'channels': channels},
            'damaged_segments': self.get_damaged_labels(board_path, array.shape),
            'skipped_segments': [{'row': int(r), 'col': int(c)} for (r, c) in sorted(skipped_segments)],
            'inspection': inspection.as_payload()
        }

        return payload

    def segment_image_payload(self, board_path: Path, row: int, col: int) -> Dict[str, object]:
        array = self._load_board(board_path)
        H, W, _, _, _ = array.shape
        if not (0 <= row < H and 0 <= col < W):
            raise ValueError(f"Segment index out of range: ({row}, {col}) within ({H}, {W})")

        segment = array[row, col]
        image_str = encode_png(segment)

        return {
            'board_path': repo_relative(board_path),
            'row': int(row),
            'col': int(col),
            'image': image_str,
        }

    def run_inspection_job(self, board_path: Path) -> subprocess.CompletedProcess:
        script_path = to_wsl_path(REPO_ROOT / 'jobs' / 'main.sh')
        board_arg = to_wsl_path(board_path)
        process = subprocess.run(
            ['bash', script_path, board_arg],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )

        if process.returncode == 0:
            self.get_inspection(board_path, force=True)

        return process