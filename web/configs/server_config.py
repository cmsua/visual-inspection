from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path('.')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INTERFACE_DIR = REPO_ROOT / 'web' / 'interface'
DAMAGED_SEGMENTS_PATH = REPO_ROOT / 'calibrations' / 'damaged_segments.json'
SKIPPED_SEGMENTS_PATH = REPO_ROOT / 'calibrations' / 'skipped_segments.json'
AE_THRESHOLD_PATH = REPO_ROOT / 'calibrations' / 'ae_threshold.npy'
PW_THRESHOLD_PATH = REPO_ROOT / 'calibrations' / 'pw_threshold.npy'
BASELINE_BOARD_PATH = REPO_ROOT / 'data' / 'train' / 'aligned_images1.npy'
DEFAULT_BOARD_PATH = REPO_ROOT / 'data' / 'bad_example' / 'aligned_images5.npy'

MAX_DISPLAY_PX = 768