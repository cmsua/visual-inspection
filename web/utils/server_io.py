from __future__ import annotations

import base64
from pathlib import Path, PureWindowsPath

import numpy as np

from ..configs import MAX_DISPLAY_PX, REPO_ROOT
from src.utils.data import array_to_pil, fit_image_to_max_side, pil_to_png_bytes


def repo_relative(path: Path) -> str:
    """Return a repository-relative path string for display/storage."""
    try:
        return f"./{path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()}"
    except ValueError:
        return path.as_posix()


def encode_png(image: np.ndarray) -> str:
    """Convert a segment array into a base64-encoded PNG string."""
    pil_img = array_to_pil(image)
    pil_img = fit_image_to_max_side(pil_img, MAX_DISPLAY_PX)
    png_bytes = pil_to_png_bytes(pil_img)

    return base64.b64encode(png_bytes).decode('ascii')


def to_wsl_path(path: Path) -> str:
    """
    Convert a filesystem path to a form understood by bash on Windows+WSL.

    If the path is already a POSIX path (e.g., /mnt/c/...), it is returned as-is.
    Otherwise, drive-letter paths such as ``C:\\Users\\...`` become ``/mnt/c/...``.
    """
    resolved = Path(path).resolve()
    posix = resolved.as_posix()
    if posix.startswith('/'):
        return posix

    drive = resolved.drive or ''
    if drive:
        win_path = PureWindowsPath(resolved)
        parts = [part for part in win_path.parts if part != win_path.drive]
        wsl_path = Path('/mnt', drive.rstrip(':').lower(), *parts)
        
        return wsl_path.as_posix()

    return posix