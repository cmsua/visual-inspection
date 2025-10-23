#!/usr/bin/env python3
"""
Local inspection web server.

Serves the static HTML/CSS/JS interface from ``web/interface`` and exposes a
small JSON API for:
    - loading board metadata and inspection flags (via InspectionResults)
    - streaming individual segment images as PNG data URLs
    - triggering the inspection pipeline through jobs/main.sh
    - persisting manual damaged labels using the existing JSONStore logic
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

# Ensure the repository root is on the Python path so imports from src/ work.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import inspect as inspect_script  # noqa: E402
from src.utils.data import load_hexaboard, load_skipped_segments  # noqa: E402
from src.utils.data.segment_labeler import (  # noqa: E402
    JSONStore,
    array_to_pil,
    fit_image_to_max_side,
    pil_to_png_bytes,
)
from src.utils.get_results import InspectionResults  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration constants

INTERFACE_DIR = REPO_ROOT / "web" / "interface"
DAMAGED_SEGMENTS_PATH = REPO_ROOT / "calibrations" / "damaged_segments.json"
SKIPPED_SEGMENTS_PATH = REPO_ROOT / "calibrations" / "skipped_segments.json"
AE_THRESHOLD_PATH = REPO_ROOT / "calibrations" / "ae_threshold.npy"
PW_THRESHOLD_PATH = REPO_ROOT / "calibrations" / "pw_threshold.npy"
BASELINE_BOARD_PATH = REPO_ROOT / "data" / "train" / "aligned_images1.npy"
DEFAULT_BOARD_PATH = REPO_ROOT / "data" / "bad_example" / "aligned_images5.npy"

MAX_DISPLAY_PX = 768

# --------------------------------------------------------------------------- #
# Utilities and data containers


def repo_relative(path: Path) -> str:
    """Return a repository-relative path string for display/storage."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
        return f"./{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def encode_png(image: np.ndarray) -> str:
    """Convert a segment array into a base64-encoded PNG string."""
    pil_img = array_to_pil(image)
    pil_img = fit_image_to_max_side(pil_img, MAX_DISPLAY_PX)
    png_bytes = pil_to_png_bytes(pil_img)
    return base64.b64encode(png_bytes).decode("ascii")


def to_wsl_path(path: Path) -> str:
    """
    Convert a filesystem path to a form understood by bash on Windows+WSL.

    If the path is already a POSIX path (e.g., /mnt/c/...), it is returned as-is.
    Otherwise, drive-letter paths such as ``C:\\Users\\...`` become ``/mnt/c/...``.
    """
    resolved = Path(path).resolve()
    posix = resolved.as_posix()
    if posix.startswith("/"):
        # POSIX/WSL path already.
        return posix

    drive = resolved.drive or ""
    if drive:
        win_path = PureWindowsPath(resolved)
        parts = [part for part in win_path.parts if part != win_path.drive]
        wsl_path = Path("/mnt", drive.rstrip(":").lower(), *parts)
        return wsl_path.as_posix()

    return posix


@dataclass
class CachedInspection:
    board_path: Path
    baseline_path: Path
    inspection: InspectionResults
    board_mtime: float
    ae_threshold_mtime: float
    pw_threshold_mtime: float
    skipped_mtime: float

    def as_payload(self) -> Dict[str, object]:
        return {
            "pw_flags": self.inspection.pw_flags.tolist(),
            "ae_flags": self.inspection.ae_flags.tolist(),
            "hybrid_flags": self.inspection.hybrid_flags.tolist(),
            "metadata": self.inspection.metadata,
            "baseline_path": repo_relative(self.baseline_path),
            "inspected_path": repo_relative(self.board_path),
        }


class InspectionApp:
    """Stateful helper attached to the HTTP server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._json_store = JSONStore(str(DAMAGED_SEGMENTS_PATH))
        self._board_cache: Dict[Path, Dict[str, object]] = {}
        self._inspection_cache: Dict[Path, CachedInspection] = {}

    # -------------------------- Board loading -------------------------------- #

    def _load_board(self, board_path: Path) -> np.ndarray:
        with self._lock:
            cached = self._board_cache.get(board_path)
            mtime = board_path.stat().st_mtime
            if cached and cached["mtime"] >= mtime:
                return cached["array"]

            array = load_hexaboard(str(board_path), normalize=False)
            self._board_cache[board_path] = {"array": array, "mtime": mtime}
            return array

    @staticmethod
    def _board_identifier(board_path: Path) -> str:
        return repo_relative(board_path)

    # ------------------------ Inspection results ----------------------------- #

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

    def get_inspection(self, board_path: Path, force: bool = False) -> CachedInspection:
        with self._lock:
            if force or self._needs_refresh(board_path):
                inspection = inspect_script.main(
                    baseline_hexaboard_path=str(BASELINE_BOARD_PATH),
                    new_hexaboard_path=str(board_path),
                    skipped_segments_path=str(SKIPPED_SEGMENTS_PATH),
                    ae_threshold_path=str(AE_THRESHOLD_PATH),
                    pw_threshold_path=str(PW_THRESHOLD_PATH),
                )
                cache_entry = CachedInspection(
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

    # -------------------------- Damaged labels -------------------------------- #

    def get_damaged_labels(self, board_path: Path, shape: Tuple[int, int, int, int, int]) -> List[Dict[str, int]]:
        board_id = self._board_identifier(board_path)
        self._json_store.ensure_file_entry(board_id, shape)
        damaged = self._json_store.get_damaged_set(board_id)
        return [{"row": int(r), "col": int(c)} for (r, c) in sorted(damaged)]

    def update_damaged_label(self, board_path: Path, shape: Tuple[int, int, int, int, int], row: int, col: int, is_damaged: bool) -> None:
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

    # ----------------------------- API helpers -------------------------------- #

    def build_board_payload(self, board_path: Path) -> Dict[str, object]:
        array = self._load_board(board_path)
        H, W, height, width, channels = array.shape
        inspection = self.get_inspection(board_path)
        skipped_segments = load_skipped_segments(str(SKIPPED_SEGMENTS_PATH))

        payload = {
            "board_path": repo_relative(board_path),
            "baseline_path": repo_relative(BASELINE_BOARD_PATH),
            "grid_shape": {"rows": H, "cols": W},
            "segment_shape": {"height": height, "width": width, "channels": channels},
            "damaged_segments": self.get_damaged_labels(board_path, array.shape),
            "skipped_segments": [{"row": int(r), "col": int(c)} for (r, c) in sorted(skipped_segments)],
            "inspection": inspection.as_payload(),
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
            "board_path": repo_relative(board_path),
            "row": int(row),
            "col": int(col),
            "image": image_str,
        }

    def run_inspection_job(self, board_path: Path) -> subprocess.CompletedProcess:
        script_path = to_wsl_path(REPO_ROOT / "jobs" / "main.sh")
        board_arg = to_wsl_path(board_path)
        process = subprocess.run(
            ["bash", script_path, board_arg],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )

        # Force inspection cache refresh on success
        if process.returncode == 0:
            self.get_inspection(board_path, force=True)

        return process


# --------------------------------------------------------------------------- #
# HTTP handler


class InspectionRequestHandler(SimpleHTTPRequestHandler):
    server_version = "HexaInspection/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(INTERFACE_DIR), **kwargs)

    @property
    def app(self) -> InspectionApp:
        return self.server.app  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ------------------------------- GET ------------------------------------ #

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            try:
                self.handle_api_get(parsed.path, parse_qs(parsed.query))
            except FileNotFoundError:
                self._send_json({"error": "Board not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pylint: disable=broad-except
                self.log_error("GET %s failed: %s", parsed.path, exc)
                self._send_json({"error": "Internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        super().do_GET()

    def handle_api_get(self, path: str, params: Dict[str, List[str]]) -> None:
        if path == "/api/board":
            board_param = params.get("path", [repo_relative(DEFAULT_BOARD_PATH)])[0]
            board_path = self._resolve_board_path(board_param)
            payload = self.app.build_board_payload(board_path)
            self._send_json(payload)
            return

        if path == "/api/segment":
            board_param = params.get("path", [repo_relative(DEFAULT_BOARD_PATH)])[0]
            row = int(params.get("row", [0])[0])
            col = int(params.get("col", [0])[0])
            board_path = self._resolve_board_path(board_param)
            payload = self.app.segment_image_payload(board_path, row=row, col=col)
            self._send_json(payload)
            return

        self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

    # ------------------------------- POST ----------------------------------- #

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8")) if body else {}
                self.handle_api_post(parsed.path, payload)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            except FileNotFoundError:
                self._send_json({"error": "Board not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except subprocess.CalledProcessError as exc:
                self.log_error("Process failed: %s", exc)
                self._send_json({"error": "Inspection run failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception as exc:  # pylint: disable=broad-except
                self.log_error("POST %s failed: %s", parsed.path, exc)
                self._send_json({"error": "Internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_api_post(self, path: str, payload: Dict[str, object]) -> None:
        if path == "/api/label":
            board_param = str(payload.get("path") or repo_relative(DEFAULT_BOARD_PATH))
            row = int(payload["row"])
            col = int(payload["col"])
            is_damaged = bool(payload.get("damaged", False))
            board_path = self._resolve_board_path(board_param)
            array = self.app._load_board(board_path)  # pylint: disable=protected-access
            self.app.update_damaged_label(board_path, array.shape, row=row, col=col, is_damaged=is_damaged)
            updated = self.app.get_damaged_labels(board_path, array.shape)
            self._send_json({"damaged_segments": updated})
            return

        if path == "/api/run":
            board_param = str(payload.get("path") or repo_relative(DEFAULT_BOARD_PATH))
            board_path = self._resolve_board_path(board_param)
            result = self.app.run_inspection_job(board_path)
            status = "ok" if result.returncode == 0 else "error"
            self._send_json(
                {
                    "status": status,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                status=HTTPStatus.OK if result.returncode == 0 else HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

    # ------------------------------- Helpers -------------------------------- #

    def _resolve_board_path(self, board_param: str) -> Path:
        if not board_param:
            return DEFAULT_BOARD_PATH

        candidate = (REPO_ROOT / board_param).resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)

        return candidate

    def _send_json(self, payload: Dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


# --------------------------------------------------------------------------- #
# Entry point


def run_server(host: str = "127.0.0.1", port: int = 3000) -> None:
    if not INTERFACE_DIR.exists():
        raise RuntimeError(f"Frontend directory not found: {INTERFACE_DIR}")

    server = ThreadingHTTPServer((host, port), InspectionRequestHandler)
    server.app = InspectionApp()  # type: ignore[attr-defined]
    print(f"Serving inspection UI at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
