from typing import Callable, Optional, Union
from pathlib import Path

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .hexaboard_loader import load_hexaboard, load_skipped_segments


class HexaboardDataset(Dataset):
    """
    Dataset of hexaboard segments for a CNN autoencoder.
    
    Similar to torchvision's ImageFolder, this class loads hexaboard data from a directory
    containing multiple .npy files, where each file represents one hexaboard.

    Expected file structure:

    root/
        ├── board_001.npy  # (H_seg, V_seg, height, width, num_channels)
        ├── board_002.npy
        └── ...

    Parameters
    ----------
    root : str or Path
        Root directory path containing .npy files, or a single .npy file path for backward compatibility.
    skipped_segments_path: str, optional
        Path to the JSON file containing the list of segments to skip.
    transform : Callable, optional
        A function/transformation that takes in a numpy array of shape
        (height, width, num_channels) and returns a torch.Tensor of shape (num_channels, height, width).
        E.g. transforms.ToTensor().

    Returns
    -------
    Tensor
        A tensor of shape (num_channels, height, width) for each segment.
        E.g. if each board has shape (12, 9, 1016, 1640, 3), then each segment will be of shape (3, 1016, 1640).
    """
    def __init__(
        self,
        root: Union[str, Path],
        skipped_segments_path: Optional[str] = None,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.transform = transform
        self.root = Path(root)

        # Find all .npy files in the directory
        self.file_paths = sorted(list(self.root.glob('*.npy')))
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npy files found in directory {self.root}")

        # Load the first file to get dimensions
        first_board = np.load(self.file_paths[0])
        if first_board.ndim != 5:
            raise ValueError(f"Expected 5D arrays (H_seg, V_seg, height, width, num_channels), got shape {first_board.shape}")
        
        self.h_seg, self.v_seg, self.height, self.width, self.num_channels = first_board.shape
        self.num_boards = len(self.file_paths)

        # Default skip set
        default_skipped = {
            (0, 0), (0, 1), (0, 7), (0, 8),
            (1, 0), (1, 8),
            (2, 0), (2, 8),
            (3, 0), (3, 8),
            (4, 0), (4, 8),
            (8, 0), (8, 8),
            (9, 0), (9, 8),
            (10, 0), (10, 1), (10, 7), (10, 8),
            (11, 0), (11, 1), (11, 8),
            (12, 0), (12, 1), (12, 7), (12, 8)
        }
        self.skipped_segments = load_skipped_segments(skipped_segments_path) if skipped_segments_path is not None else default_skipped
        
        # Keep only valid indices inside the board shape
        self.skipped_segments = {
            (h, v) for (h, v) in self.skipped_segments if (0 <= h < self.h_seg and 0 <= v < self.v_seg)
        }

        # Precompute valid (h, v) segment indices per board
        self.valid_segments = [
            (h, v) for h in range(self.h_seg)
            for v in range(self.v_seg)
            if (h, v) not in self.skipped_segments
        ]

        if len(self.valid_segments) == 0:
            raise ValueError("All segments are skipped; no data to load.")

        self.segs_per_board = len(self.valid_segments)
        self.total = self.num_boards * self.segs_per_board

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tensor:
        # Calculate which board and which segment within that board
        board_idx = idx // self.segs_per_board
        seg_idx = idx % self.segs_per_board
        h_idx, v_idx = self.valid_segments[seg_idx]
        
        # Load the hexaboard data and extract the segment
        hexaboard = load_hexaboard(self.file_paths[board_idx], normalize=False)
        segment = hexaboard[h_idx, v_idx]

        if self.transform:
            tensor = self.transform(segment)
        else:
            tensor = torch.from_numpy(segment).permute(2, 0, 1).float()

        return tensor