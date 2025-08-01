from typing import Callable, Optional, Union
from pathlib import Path

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


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
    root : Union[str, Path]
        Root directory path containing .npy files, or a single .npy file path for backward compatibility.
    transform : Optional[Callable]
        A function/transformation that takes in a numpy array of shape
        (height, width, num_channels) and returns a torch.Tensor of shape (num_channels, height, width).
        E.g. transforms.ToTensor().

    Returns
    -------
    Tensor
        A tensor of shape (num_channels, height, width) for each segment.
        E.g. if each board has shape (12, 9, 1016, 1640, 3), then each segment will be of shape (3, 1016, 1640).
    """
    def __init__(self, root: Union[str, Path], transform: Optional[Callable] = None):
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
        
        self.total = self.num_boards * self.h_seg * self.v_seg
        self.segs_per_board = self.h_seg * self.v_seg

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tensor:
        # Calculate which board and which segment within that board
        board_idx = idx // self.segs_per_board
        remainder = idx % self.segs_per_board
        h_idx = remainder // self.v_seg
        v_idx = remainder % self.v_seg
        
        # Load the board data
        board_data = np.load(self.file_paths[board_idx])
        segment = board_data[h_idx, v_idx]

        if self.transform:
            tensor = self.transform(segment)
        else:
            tensor = torch.from_numpy(segment).permute(2, 0, 1).float()

        return tensor