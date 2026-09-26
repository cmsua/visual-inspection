from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .data.board_contract import validate_coordinates

SegmentIndex = Tuple[int, int]

# Segment status encoding understood by the vanilla JavaScript frontend
SKIPPED_SEGMENT = -1
OK_SEGMENT = 0
FLAGGED_SEGMENT = 1


@dataclass
class InspectionResults:
    """
    Structured payload shared between the Python inspection pipeline and the vanilla JavaScript UI.

    Attributes
    ----------
    pw_flags : np.ndarray
        2D array with segment statuses for the pixel-wise comparison method.
    ae_flags : np.ndarray
        2D array with segment statuses for the autoencoder method.
    hybrid_flags : np.ndarray
        2D array showing segments flagged by both methods.
    baseline_path : str
        Path to the baseline hexaboard `.npy` file.
    inspected_path : str
        Path to the inspected hexaboard `.npy` file.
    metadata : Dict[str, object]
        Additional context such as counts, segment shapes, and status encoding.
    """

    pw_flags: np.ndarray
    ae_flags: np.ndarray
    hybrid_flags: np.ndarray
    baseline_path: str
    inspected_path: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def _build_flag_grid(
        shape: Tuple[int, int],
        flagged: Iterable[SegmentIndex],
        skipped: Iterable[SegmentIndex]
    ) -> np.ndarray:
        grid = np.full(shape, OK_SEGMENT, dtype=np.int8)
        skipped_set = validate_coordinates(skipped, shape)
        flagged = validate_coordinates(flagged, shape)
        for h, v in skipped_set:
            grid[h, v] = SKIPPED_SEGMENT

        for h, v in flagged:
            if (h, v) in skipped_set:
                continue

            grid[h, v] = FLAGGED_SEGMENT

        return grid

    @staticmethod
    def summarize_flagged_segments(
        pixel_flagged: Iterable[SegmentIndex],
        autoencoder_flagged: Iterable[SegmentIndex]
    ) -> Dict[str, List[SegmentIndex]]:
        pixel_set = set(pixel_flagged)
        auto_set = set(autoencoder_flagged)

        both = sorted(pixel_set & auto_set)
        pixel_only = sorted(pixel_set - auto_set)
        auto_only = sorted(auto_set - pixel_set)

        return {'pixel_only': pixel_only, 'autoencoder_only': auto_only, 'both_methods': both}

    @classmethod
    def from_segment_flags(
        cls,
        shape: Tuple[int, int],
        pixel_flagged: Iterable[SegmentIndex],
        autoencoder_flagged: Iterable[SegmentIndex],
        skipped_segments: Set[SegmentIndex],
        baseline_path: str,
        inspected_path: str,
        segment_shape: Optional[Tuple[int, int, int]] = None
    ) -> 'InspectionResults':
        pixel_flagged = set(pixel_flagged)
        autoencoder_flagged = set(autoencoder_flagged)
        pw_flags = cls._build_flag_grid(shape, pixel_flagged, skipped_segments)
        ae_flags = cls._build_flag_grid(shape, autoencoder_flagged, skipped_segments)
        hybrid_indices = set(pixel_flagged) & set(autoencoder_flagged)
        hybrid_flags = cls._build_flag_grid(shape, hybrid_indices, skipped_segments)
        flagged_segments = cls.summarize_flagged_segments(pixel_flagged, autoencoder_flagged)

        metadata = {
            'board_shape': list(shape),
            'segment_shape': list(segment_shape) if segment_shape else None,
            'status_encoding': {'skipped': SKIPPED_SEGMENT, 'ok': OK_SEGMENT, 'flagged': FLAGGED_SEGMENT},
            'counts': {
                'pixel_flagged': int(np.sum(pw_flags == FLAGGED_SEGMENT)),
                'autoencoder_flagged': int(np.sum(ae_flags == FLAGGED_SEGMENT)),
                'hybrid_flagged': int(np.sum(hybrid_flags == FLAGGED_SEGMENT)),
                'skipped': int(np.sum(pw_flags == SKIPPED_SEGMENT))
            },
            'flagged_segments': flagged_segments
        }

        return cls(
            pw_flags=pw_flags,
            ae_flags=ae_flags,
            hybrid_flags=hybrid_flags,
            baseline_path=baseline_path,
            inspected_path=inspected_path,
            metadata=metadata
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            'pw_flags': self.pw_flags.tolist(),
            'ae_flags': self.ae_flags.tolist(),
            'hybrid_flags': self.hybrid_flags.tolist(),
            'baseline_path': self.baseline_path,
            'inspected_path': self.inspected_path,
            'metadata': self.metadata
        }

    def save_json(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(self.to_dict(), handle, indent=4, allow_nan=False)

            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load_json(cls, path: str) -> 'InspectionResults':
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)

        grids = [np.asarray(payload[key]) for key in ('pw_flags', 'ae_flags', 'hybrid_flags')]
        shape = tuple(payload['metadata']['board_shape'])
        if len(shape) != 2 or any(grid.shape != shape or not np.isin(grid, [-1, 0, 1]).all() for grid in grids):
            raise ValueError("Invalid persisted inspection grids.")

        return cls(*grids, payload['baseline_path'], payload['inspected_path'], payload['metadata'])

    def summary(self) -> Dict[str, int]:
        return {
            'pixel_flagged': int(np.sum(self.pw_flags == FLAGGED_SEGMENT)),
            'autoencoder_flagged': int(np.sum(self.ae_flags == FLAGGED_SEGMENT)),
            'hybrid_flagged': int(np.sum(self.hybrid_flags == FLAGGED_SEGMENT)),
            'skipped': int(np.sum(self.pw_flags == SKIPPED_SEGMENT))
        }
