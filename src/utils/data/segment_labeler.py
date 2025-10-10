"""Interactive segment labeling utilities.

This module provides the building blocks for annotating hexaboard
segments both inside Jupyter notebooks and from a future web frontend.
The design separates three main responsibilities:

* ``LabelStore`` persists user decisions and checkpoints to JSON.
* ``SegmentLabelerSession`` encapsulates iteration and state handling
  for a concrete hexaboard array.
* ``build_notebook_widget`` offers a lightweight ipywidgets UI that can
  be reused in notebooks without conflicting with a future React app.

The exported helpers intentionally keep their I/O formats simple so a
web client can mirror the behaviour by reproducing the same JSON
contract defined in :class:`LabelStore`.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import ipywidgets as widgets


Array5D = np.ndarray
SegmentCoordinates = Tuple[int, int]


class LabelState(str, Enum):
	"""Enumeration for segment annotation states."""

	UNLABELED = "unlabeled"
	DAMAGED = "damaged"
	GOOD = "good"


@dataclass(frozen=True)
class SegmentIndex:
	"""Row/column index of a segment within a hexaboard."""

	row: int
	col: int

	def to_tuple(self) -> SegmentCoordinates:
		return self.row, self.col


class LabelStore:
	"""Persist segment labels and checkpoints to a JSON file.

	The JSON schema is intentionally simple:

	.. code-block:: json

		{
		  "version": 1,
		  "files": {
			"<file_id>": {
			  "file_id": "<file_id>",
			  "shape": [R, C, H, W, Ch],
			  "labels": [
				{"row": 0, "col": 0, "state": "damaged"},
				...
			  ]
			}
		  },
		  "checkpoints": {
			"<file_id>": {"row": 0, "col": 1, "index": 5}
		  }
		}

	Future frontends can rely on this structure to stay compatible with
	the notebook tooling.
	"""

	VERSION = 1

	def __init__(self, path: Union[str, Path]):
		self.path = Path(path)
		self._data: Dict[str, Any] = {
			"version": self.VERSION,
			"files": {},
			"checkpoints": {},
		}
		self._load()

	# ------------------------------------------------------------------
	# Loading / saving
	# ------------------------------------------------------------------
	def _load(self) -> None:
		if self.path.is_file():
			try:
				text = self.path.read_text(encoding="utf-8")
				self._data = json.loads(text)
			except json.JSONDecodeError:
				backup = self.path.with_suffix(self.path.suffix + ".bak")
				try:
					os.replace(self.path, backup)
					print(
						f"[WARN] Damaged JSON detected. Backed up to {backup} and "
						"created a new store."
					)
				except OSError:
					print(
						"[WARN] Damaged JSON detected and could not be backed up. "
						"Starting fresh."
					)
				self._data = {
					"version": self.VERSION,
					"files": {},
					"checkpoints": {},
				}

		self._data.setdefault("version", self.VERSION)
		self._data.setdefault("files", {})
		self._data.setdefault("checkpoints", {})

	def _atomic_write(self) -> None:
		target_dir = self.path.parent
		if target_dir:
			target_dir.mkdir(parents=True, exist_ok=True)

		fd, tmp_path = tempfile.mkstemp(prefix=".tmp_label_store_", suffix=".json", dir=target_dir)
		try:
			with os.fdopen(fd, "w", encoding="utf-8") as handle:
				json.dump(self._data, handle, indent=2, sort_keys=True)
			os.replace(tmp_path, self.path)
		finally:
			try:
				if os.path.exists(tmp_path):
					os.remove(tmp_path)
			except OSError:
				pass

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def ensure_file_entry(self, file_id: str, shape: Sequence[int]) -> None:
		files = self._data.setdefault("files", {})
		entry = files.get(file_id)
		if entry is None:
			files[file_id] = {
				"file_id": file_id,
				"shape": list(shape),
				"labels": [],
			}
		else:
			if "shape" not in entry:
				entry["shape"] = list(shape)
			elif tuple(entry.get("shape", ())) != tuple(shape):
				print(
					f"[WARN] Existing annotation shape {entry.get('shape')} "
					f"differs from current {list(shape)}. Updating metadata."
				)
				entry["shape"] = list(shape)
			entry.setdefault("labels", [])
		self._atomic_write()

	def load_labels(self, file_id: str) -> Dict[SegmentIndex, LabelState]:
		entry = self._data.get("files", {}).get(file_id)
		if not entry:
			return {}

		labels: Dict[SegmentIndex, LabelState] = {}
		for item in entry.get("labels", []):
			try:
				row = int(item["row"])
				col = int(item["col"])
				state = LabelState(item.get("state", LabelState.DAMAGED))
			except (KeyError, ValueError):
				continue
			labels[SegmentIndex(row=row, col=col)] = state
		return labels

	def save_labels(self, file_id: str, labels: Dict[SegmentIndex, LabelState]) -> None:
		serialized = [
			{"row": idx.row, "col": idx.col, "state": state.value}
			for idx, state in sorted(labels.items(), key=lambda item: (item[0].row, item[0].col))
			if state != LabelState.UNLABELED
		]

		files = self._data.setdefault("files", {})
		entry = files.setdefault(file_id, {"file_id": file_id, "labels": []})
		entry["labels"] = serialized
		self._atomic_write()

	def load_checkpoint(self, file_id: str) -> Optional[Dict[str, int]]:
		checkpoint = self._data.get("checkpoints", {}).get(file_id)
		if checkpoint is None:
			return None
		try:
			return {
				"row": int(checkpoint["row"]),
				"col": int(checkpoint["col"]),
				"index": int(checkpoint["index"]),
			}
		except (KeyError, ValueError):
			return None

	def save_checkpoint(self, file_id: str, row: int, col: int, index: int) -> None:
		checkpoints = self._data.setdefault("checkpoints", {})
		checkpoints[file_id] = {"row": int(row), "col": int(col), "index": int(index)}
		self._atomic_write()


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
	if array.dtype == np.uint8:
		return array

	finite_mask = np.isfinite(array)
	if not finite_mask.any():
		return np.zeros_like(array, dtype=np.uint8)

	arr = array.astype(np.float32, copy=False)
	vmin = float(np.nanmin(arr))
	vmax = float(np.nanmax(arr))
	if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
		return np.zeros_like(arr, dtype=np.uint8)

	scaled = (arr - vmin) / (vmax - vmin)
	scaled = np.clip(scaled * 255.0, 0.0, 255.0)
	return scaled.astype(np.uint8)


def _array_to_rgb(array: np.ndarray) -> np.ndarray:
	if array.ndim == 2:
		arr = _normalize_to_uint8(array)
		return np.repeat(arr[..., None], 3, axis=-1)

	if array.ndim == 3:
		channels = array.shape[-1]
		if channels == 3:
			return _normalize_to_uint8(array)
		if channels >= 3:
			return _normalize_to_uint8(array[..., :3])
		if channels == 1:
			arr = _normalize_to_uint8(array[..., 0])
			return np.repeat(arr[..., None], 3, axis=-1)

	raise ValueError(f"Unsupported image shape {array.shape} for RGB conversion.")


def _load_array_from_file(
	input_path: Union[str, Path],
	*,
	key: Optional[str] = None,
	mmap: bool = True,
	normalize: bool = False,
) -> Array5D:
	path = Path(input_path).expanduser()
	if not path.exists():
		raise FileNotFoundError(f"Input file not found: {path}")

	suffix = path.suffix.lower()
	if suffix == ".npy":
		mmap_mode = "r" if mmap else None
		arr = np.load(path, mmap_mode=mmap_mode)
	elif suffix == ".npz":
		if key is None:
			raise ValueError("For .npz inputs, a 'key' must be provided.")
		with np.load(path) as archive:
			if key not in archive:
				raise KeyError(f"Key '{key}' not present in archive {path}.")
			arr = archive[key]
	else:
		raise ValueError(f"Unsupported file extension '{suffix}'. Use .npy or .npz.")

	if arr.ndim != 5:
		raise ValueError(
			f"Expected a 5D array (rows, cols, height, width, channels); got shape {arr.shape}."
		)

	array = np.asarray(arr[..., ::-1])  # BGR -> RGB
	if normalize:
		array = array.astype(np.float32, copy=False) / 255.0
	else:
		array = np.array(array, copy=True)

	return array


class SegmentLabelerSession:
	"""Mutable state for annotating a single hexaboard array."""

	def __init__(
		self,
		array: Array5D,
		*,
		file_id: str,
		store: LabelStore,
		skip_segments: Optional[Iterable[SegmentCoordinates]] = None,
		start_row: Optional[int] = None,
		start_col: Optional[int] = None,
	):
		if array.ndim != 5:
			raise ValueError(
				f"Expected a 5D array (rows, cols, height, width, channels); got shape {array.shape}."
			)

		self.array = array
		self.file_id = file_id
		self.store = store
		self.rows, self.cols, self.height, self.width, self.channels = array.shape

		self.store.ensure_file_entry(file_id, (self.rows, self.cols, self.height, self.width, self.channels))

		skipped = {SegmentIndex(*coords) for coords in (skip_segments or [])}
		self.indices: List[SegmentIndex] = [
			SegmentIndex(row=row, col=col)
			for row in range(self.rows)
			for col in range(self.cols)
			if SegmentIndex(row=row, col=col) not in skipped
		]

		if not self.indices:
			raise ValueError("No segments available after applying skip list.")

		self.labels: Dict[SegmentIndex, LabelState] = self.store.load_labels(file_id)

		checkpoint = self.store.load_checkpoint(file_id)
		if start_row is not None and start_col is not None:
			cursor = self._cursor_from_rc(start_row, start_col)
		elif checkpoint is not None:
			cursor = int(checkpoint.get("index", 0))
		else:
			cursor = 0

		self.cursor = max(0, min(cursor, len(self.indices) - 1))

	# ------------------------------------------------------------------
	# Navigation helpers
	# ------------------------------------------------------------------
	def _cursor_from_rc(self, row: int, col: int) -> int:
		target = SegmentIndex(row=row, col=col)
		try:
			return self.indices.index(target)
		except ValueError:
			return 0

	@property
	def current_index(self) -> SegmentIndex:
		return self.indices[self.cursor]

	def goto(self, row: int, col: int) -> SegmentIndex:
		self.cursor = self._cursor_from_rc(row, col)
		self._autosave_checkpoint()
		return self.current_index

	def next(self) -> SegmentIndex:
		self.cursor = (self.cursor + 1) % len(self.indices)
		self._autosave_checkpoint()
		return self.current_index

	def prev(self) -> SegmentIndex:
		self.cursor = (self.cursor - 1) % len(self.indices)
		self._autosave_checkpoint()
		return self.current_index

	# ------------------------------------------------------------------
	# Label management
	# ------------------------------------------------------------------
	def label_current(self, state: LabelState) -> None:
		if state == LabelState.UNLABELED:
			self.labels.pop(self.current_index, None)
		else:
			self.labels[self.current_index] = state
		self.store.save_labels(self.file_id, self.labels)

	def mark_damaged(self) -> None:
		self.label_current(LabelState.DAMAGED)
		self.next()

	def mark_undamaged(self) -> None:
		self.label_current(LabelState.GOOD)
		self.next()

	# ------------------------------------------------------------------
	# Data access
	# ------------------------------------------------------------------
	def get_segment(self, index: Optional[SegmentIndex] = None) -> np.ndarray:
		idx = index or self.current_index
		return np.array(self.array[idx.row, idx.col])

	def current_label(self) -> LabelState:
		return self.labels.get(self.current_index, LabelState.UNLABELED)

	def progress(self) -> Dict[str, Any]:
		labeled = sum(1 for state in self.labels.values() if state != LabelState.UNLABELED)
		return {
			"file_id": self.file_id,
			"shape": {
				"rows": self.rows,
				"cols": self.cols,
				"height": self.height,
				"width": self.width,
				"channels": self.channels,
			},
			"cursor": self.cursor,
			"current": {
				"row": self.current_index.row,
				"col": self.current_index.col,
				"state": self.current_label().value,
			},
			"total_segments": len(self.indices),
			"labeled_segments": labeled,
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _autosave_checkpoint(self) -> None:
		current = self.current_index
		self.store.save_checkpoint(self.file_id, current.row, current.col, self.cursor)


def build_notebook_widget(
	session: SegmentLabelerSession,
	*,
	max_display_px: int = 768,
) -> "widgets.VBox":
	"""Create an ipywidgets UI bound to the provided session.

	The function imports ipywidgets lazily so the module can be used in
	non-notebook contexts without pulling extra dependencies.
	"""

	try:
		import ipywidgets as widgets
	except Exception as exc:  # pragma: no cover - optional dependency
		raise ImportError(
			"ipywidgets is required for the notebook UI. Install with 'pip install ipywidgets'."
		) from exc

	from PIL import Image

	def _fit_image(image: Image.Image, max_side: int) -> Image.Image:
		if max_side <= 0:
			return image
		width, height = image.size
		scale = min(max_side / float(width), max_side / float(height), 1.0)
		new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
		if new_size == image.size:
			return image
		return image.resize(new_size, resample=Image.BILINEAR)

	def _to_png_bytes(array: np.ndarray, max_side: int) -> Tuple[bytes, Tuple[int, int]]:
		rgb = _array_to_rgb(array)
		image = Image.fromarray(rgb)
		image = _fit_image(image, max_side)
		with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
			image.save(tmp.name, format="PNG")
			tmp.seek(0)
			return tmp.read(), image.size

	info_label = widgets.HTML()
	status_label = widgets.HTML()
	image_widget = widgets.Image(format="png")

	prev_btn = widgets.Button(description="⟵ Previous")
	next_btn = widgets.Button(description="Next ⟶")
	damaged_btn = widgets.Button(description="Damaged", button_style="danger")
	good_btn = widgets.Button(description="Good", button_style="success")
	skip_btn = widgets.Button(description="Skip", button_style="warning")
	save_btn = widgets.Button(description="Save checkpoint")

	row_field = widgets.BoundedIntText(description="Row", min=0, max=session.rows - 1, step=1)
	col_field = widgets.BoundedIntText(description="Col", min=0, max=session.cols - 1, step=1)
	goto_btn = widgets.Button(description="Go")
	size_slider = widgets.IntSlider(
		description="Size (px)", min=256, max=1280, step=64, value=max_display_px, continuous_update=False
	)

	def _update_view() -> None:
		state = session.current_label()
		seg = session.current_index
		png_bytes, (width, height) = _to_png_bytes(session.get_segment(), size_slider.value)

		info_label.value = (
			f"<b>File:</b> {session.file_id} &nbsp; | &nbsp; "
			f"<b>Shape:</b> ({session.rows}, {session.cols}, {session.height}, {session.width}, {session.channels})"
		)
		status_label.value = (
			f"<b>Row:</b> {seg.row} &nbsp; <b>Col:</b> {seg.col} &nbsp; | &nbsp; "
			f"<b>Status:</b> {state.value.title()} &nbsp; | &nbsp; "
			f"<b>Cursor:</b> {session.cursor + 1}/{len(session.indices)}"
		)

		row_field.value = seg.row
		col_field.value = seg.col
		image_widget.value = png_bytes
		image_widget.layout = widgets.Layout(width=f"{width}px", height=f"{height}px")

	def _mark_and_refresh(state: LabelState) -> None:
		session.label_current(state)
		session.next()
		_update_view()

	def _skip() -> None:
		session.label_current(LabelState.UNLABELED)
		session.next()
		_update_view()

	def _goto(_=None) -> None:
		session.goto(row_field.value, col_field.value)
		_update_view()

	def _save(_=None) -> None:
		idx = session.current_index
		session.store.save_checkpoint(session.file_id, idx.row, idx.col, session.cursor)
		status_label.value = status_label.value + " &nbsp; | &nbsp; <b>Checkpoint saved.</b>"

	prev_btn.on_click(lambda _btn: (session.prev(), _update_view()))
	next_btn.on_click(lambda _btn: (session.next(), _update_view()))
	damaged_btn.on_click(lambda _btn: _mark_and_refresh(LabelState.DAMAGED))
	good_btn.on_click(lambda _btn: _mark_and_refresh(LabelState.GOOD))
	skip_btn.on_click(lambda _btn: _skip())
	goto_btn.on_click(_goto)
	save_btn.on_click(_save)
	size_slider.observe(lambda change: change.get("name") == "value" and _update_view(), names="value")

	controls = widgets.HBox([
		prev_btn,
		next_btn,
		widgets.HTML("<span style='margin-left:1em'></span>"),
		damaged_btn,
		good_btn,
		skip_btn,
		widgets.HTML("<span style='margin-left:1em'></span>"),
		save_btn,
	])
	goto_controls = widgets.HBox([row_field, col_field, goto_btn, size_slider])

	container = widgets.VBox([info_label, image_widget, status_label, controls, goto_controls])
	_update_view()
	return container


def launch_notebook_labeler(
	input_path: Union[str, Path],
	*,
	json_path: Union[str, Path] = "damaged_segments.json",
	key: Optional[str] = None,
	skip_segments: Optional[Iterable[SegmentCoordinates]] = None,
	start_row: Optional[int] = None,
	start_col: Optional[int] = None,
	mmap: bool = True,
	normalize: bool = False,
	max_display_px: int = 768,
) -> "widgets.VBox":
	"""High-level helper mirroring the previous notebook entry point."""

	array = _load_array_from_file(input_path, key=key, mmap=mmap, normalize=normalize)
	store = LabelStore(json_path)
	session = SegmentLabelerSession(
		array,
		file_id=str(Path(input_path).name),
		store=store,
		skip_segments=skip_segments,
		start_row=start_row,
		start_col=start_col,
	)
	return build_notebook_widget(session, max_display_px=max_display_px)


__all__ = [
	"LabelState",
	"LabelStore",
	"SegmentIndex",
	"SegmentLabelerSession",
	"build_notebook_widget",
	"launch_notebook_labeler",
]
