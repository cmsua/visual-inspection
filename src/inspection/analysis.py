import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import save_json
from .run import inspect_board, validate_calibration
from ..configs.inspection_config import InspectionConfig
from ..utils.data.board_contract import GRID_SHAPE, validate_coordinates
from ..utils.viz import plot_confusion_matrices


def analyze(config: InspectionConfig, max_boards: Optional[int] = None) -> Dict:
    """
    Evaluate held-out boards on all physically inspected positions.

    Parameters
    ----------
    config : InspectionConfig
        Inspection settings and evaluation directories.
    max_boards : int, optional
        Limit each class for a bounded integration run.

    Returns
    -------
    Dict
        Confusion matrices, evaluated board paths, and metric scope.
    """
    config.resolve()
    validate_calibration(config)
    with open(Path(config.calibration_dir) / 'manifest.json', 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)

    with open(config.labels_path, 'r', encoding='utf-8') as handle:
        entries = json.load(handle)['files']

    used_ids = {Path(path).stem for paths in manifest['sources'].values() for path in paths}
    used_ids.add(Path(config.baseline_path).stem)
    labels = {}
    for path, entry in entries.items():
        damaged = validate_coordinates([(item['row'], item['col']) for item in entry.get('damaged', [])], GRID_SHAPE)
        board_id = Path(path).stem
        if board_id in labels and labels[board_id] != damaged:
            raise ValueError(f"Conflicting labels for board `{board_id}`.")

        labels[board_id] = damaged

    matrices = {key: np.zeros((2, 2), dtype=int) for key in ('pw_flags', 'ae_flags', 'hybrid_flags')}
    evaluated = []
    seen = set(used_ids)
    for is_bad, directory in ((False, config.test_dir), (True, config.bad_dir)):
        paths = [path for path in sorted(Path(directory).glob('*.npy')) if path.stem not in seen]
        if max_boards is not None:
            paths = paths[:max_boards]

        if not paths:
            raise ValueError(f"No held-out {'bad' if is_bad else 'normal'} boards available in `{directory}`.")

        for path in paths:
            if is_bad and not labels.get(path.stem):
                raise ValueError(f"Bad evaluation board lacks annotations: `{path}`.")

            if not is_bad and labels.get(path.stem):
                raise ValueError(f"Normal evaluation board has damaged labels: `{path}`.")

            result = inspect_board(config, str(path))
            truth = np.zeros(GRID_SHAPE, dtype=bool)
            for row, col in labels.get(path.stem, set()):
                truth[row, col] = True

            for key, matrix in matrices.items():
                flags = getattr(result, key)
                valid = flags >= 0
                for actual, predicted in zip(truth[valid], flags[valid] == 1):
                    matrix[int(actual), int(predicted)] += 1

            seen.add(path.stem)
            evaluated.append(str(path))
            print(f"Analyzed {path.name}", flush=True)

    report = {
        'scope': "All non-skipped segments; board IDs disjoint from calibration",
        'matrices': {key: matrix.tolist() for key, matrix in matrices.items()},
        'boards': evaluated
    }
    destination = Path(config.output_dir)
    save_json(str(destination / 'analysis.json'), report)
    figure = plot_confusion_matrices(
        matrices['ae_flags'],
        matrices['pw_flags'],
        matrices['hybrid_flags'],
        title="Held-out boards: all inspected positions",
        savefig=str(destination / 'confusion_matrices.png'),
        show=False
    )
    plt.close(figure)
    return report
