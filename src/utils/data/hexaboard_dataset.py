from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .board_contract import BOARD_SHAPE, open_board, rgb_segment
from .hexaboard_loader import load_skipped_segments


class HexaboardDataset(Dataset):
    """
    Load RGB segments from one board or a directory of validated boards.

    Parameters
    ----------
    root : str or Path
        A NumPy file or directory containing NumPy boards.
    skipped_segments_path : str, optional
        Skip mask; defaults to the canonical JSON.
    transform : Callable, optional
        Receives an RGB uint8 segment. Without a transform, returns normalized
        float32 tensors with shape `[3, height, width]`.
    expected_shape : Tuple[int, ...], optional
        Board contract, with an explicit override for small test fixtures.
    """

    def __init__(
        self,
        root: Union[str, Path],
        skipped_segments_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        expected_shape: Tuple[int, ...] = BOARD_SHAPE
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.expected_shape = tuple(expected_shape)
        self.file_paths = [self.root] if self.root.is_file() else sorted(self.root.glob('*.npy'))
        if not self.file_paths:
            raise ValueError(f"No `.npy` boards found at `{self.root}`.")

        for path in self.file_paths:
            open_board(str(path), self.expected_shape)

        self.h_seg, self.v_seg, self.height, self.width, self.num_channels = self.expected_shape
        self.num_boards = len(self.file_paths)
        self.skipped_segments = load_skipped_segments(skipped_segments_path, self.expected_shape[:2])
        self.valid_segments = [
            (row, col) for row in range(self.h_seg) for col in range(self.v_seg)
            if (row, col) not in self.skipped_segments
        ]
        if not self.valid_segments:
            raise ValueError("All segments are skipped; no data to load.")

        self.segs_per_board = len(self.valid_segments)
        self.total = self.num_boards * self.segs_per_board

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tensor:
        if not 0 <= idx < self.total:
            raise IndexError(f"Dataset index `{idx}` is out of range.")

        board_idx, segment_idx = divmod(idx, self.segs_per_board)
        row, col = self.valid_segments[segment_idx]
        board = open_board(str(self.file_paths[board_idx]), self.expected_shape)
        segment = rgb_segment(board, row, col, normalize=False)
        if self.transform is not None:
            return self.transform(segment)

        return torch.from_numpy(segment).permute(2, 0, 1).float() / 255.0
