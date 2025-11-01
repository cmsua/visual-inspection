import io
import json
import os
import tempfile
from PIL import Image
from typing import Dict, Set, Tuple, Optional

import numpy as np
from ipywidgets import widgets


class JSONStore:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = {'version': 1, 'files': {}, 'checkpoints': {}}
        self._load()

    def _load(self):
        if os.path.isfile(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                backup = self.json_path + '.bak'
                try:
                    os.replace(self.json_path, backup)
                    print(f"[WARN] Damaged JSON detected. Backed up to {backup} and starting a new one.")
                except Exception:
                    print(f"[WARN] Damaged JSON detected. Could not back it up. Starting a new one.")
                self.data = {'version': 1, 'files': {}, 'checkpoints': {}}

        if 'version' not in self.data:
            self.data['version'] = 1
            
        self.data.setdefault('files', {})
        self.data.setdefault('checkpoints', {})

    def _atomic_write(self):
        target_dir = os.path.dirname(os.path.abspath(self.json_path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix='.tmp_damaged_', suffix='.json', dir=target_dir if target_dir else None)
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, sort_keys=True)
                
            os.replace(tmp_path, self.json_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def ensure_file_entry(self, file_id: str, shape: Tuple[int, int, int, int, int]):
        files = self.data['files']
        if file_id not in files:
            files[file_id] = {'file_id': file_id, 'shape': list(shape), 'damaged': []}
        else:
            if 'shape' not in files[file_id]:
                files[file_id]['shape'] = list(shape)
            elif tuple(files[file_id]['shape']) != tuple(shape):
                print(f"[WARN] Existing JSON has shape {files[file_id]['shape']} but current file has shape {list(shape)}. Keeping labels but updating shape.")
                files[file_id]['shape'] = list(shape)
                
        self._atomic_write()

    def get_damaged_set(self, file_id: str) -> Set[Tuple[int, int]]:
        entry = self.data['files'].get(file_id, {})
        items = entry.get('damaged', [])
        s = set()
        for obj in items:
            try:
                r = int(obj['row'])
                c = int(obj['col'])
                s.add((r, c))
            except Exception:
                continue
            
        return s

    def save_damaged_set(self, file_id: str, damaged: Set[Tuple[int, int]]):
        dlist = [{'row': int(r), 'col': int(c)} for (r, c) in sorted(damaged)]
        self.data['files'].setdefault(file_id, {})['damaged'] = dlist
        self._atomic_write()

    def get_checkpoint(self, file_id: str) -> Optional[Dict[str, int]]:
        return self.data.get('checkpoints', {}).get(file_id)

    def save_checkpoint(self, file_id: str, row: int, col: int, idx: int):
        self.data.setdefault('checkpoints', {})[file_id] = {'row': int(row), 'col': int(col), 'index': int(idx)}
        self._atomic_write()


def normalize_to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    
    a = a.astype(np.float32, copy=False)
    finite_mask = np.isfinite(a)
    if not finite_mask.any():
        return np.zeros(a.shape, dtype=np.uint8)
    
    vmin = np.nanmin(a)
    vmax = np.nanmax(a)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(a.shape, dtype=np.uint8)
    
    scaled = (a - vmin) / (vmax - vmin)
    scaled *= 255.0
    
    return np.clip(scaled, 0, 255).astype(np.uint8)


def array_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 3:
        _, _, ch = img.shape
        if ch == 1:
            arr = normalize_to_uint8(img[..., 0])
        elif ch == 3 or ch == 4:
            arr = normalize_to_uint8(img)
        else:
            arr = normalize_to_uint8(img[..., :3])

        return Image.fromarray(arr)
    elif img.ndim == 2:
        arr = normalize_to_uint8(img)

        return Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Unexpected image shape for display: {img.shape}")


def fit_image_to_max_side(pil_img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return pil_img
    
    w, h = pil_img.size
    scale = min(max_side / float(w), max_side / float(h), 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if (new_w, new_h) == (w, h):
        return pil_img
    
    return pil_img.resize((new_w, new_h), resample=Image.BILINEAR)


def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    with io.BytesIO() as bio:
        pil_img.save(bio, format="PNG")
        return bio.getvalue()


class NotebookDamagedLabeler:
    def __init__(
        self,
        array: np.ndarray,
        file_id: str,
        json_path: str,
        start_row: Optional[int] = None,
        start_col: Optional[int] = None,
        max_display_px: int = 768
    ):
        self.array = array
        self.file_id = file_id
        self.json_store = JSONStore(json_path)
        self.max_display_px = int(max_display_px)

        # Validate shape
        if self.array.ndim != 5:
            raise ValueError(f"Input array must be 5D (rows, cols, H, W, C). Got: {self.array.shape}")

        self.R, self.C, self.H, self.W, self.Ch = self.array.shape
        self.total = self.R * self.C

        # JSON init for this file
        self.json_store.ensure_file_entry(self.file_id, (self.R, self.C, self.H, self.W, self.Ch))
        self.damaged = self.json_store.get_damaged_set(self.file_id)

        # Compute starting index (row-major order)
        cp = self.json_store.get_checkpoint(self.file_id)
        if start_row is not None and start_col is not None:
            self.idx = int(start_row) * self.C + int(start_col)
        elif cp is not None:
            self.idx = int(cp.get("index", 0))
        else:
            self.idx = 0

        self.idx = max(0, min(self.total - 1, self.idx))

        # Widgets
        self.info_label = widgets.HTML()
        self.image_widget = widgets.Image(format='png')
        self.status_label = widgets.HTML()

        self.prev_btn = widgets.Button(description="<- Previous")
        self.next_btn = widgets.Button(description="Next ->")
        self.damaged_btn = widgets.Button(description="Damaged")
        self.undamaged_btn = widgets.Button(description="Undamaged")
        self.save_btn = widgets.Button(description="Save Checkpoint")

        self.row_field = widgets.BoundedIntText(description="Row", min=0, max=self.R - 1, step=1)
        self.col_field = widgets.BoundedIntText(description="Col", min=0, max=self.C - 1, step=1)
        self.go_btn = widgets.Button(description="Go")

        self.size_slider = widgets.IntSlider(
            description="Size (px)",
            min=256,
            max=1280,
            step=64,
            value=self.max_display_px,
            continuous_update=False
        )

        # Layout
        buttons_row1 = widgets.HBox([
            self.prev_btn, self.next_btn,
            widgets.HTML("<span style='margin-left:1em'></span>"),
            self.damaged_btn, self.undamaged_btn,
            widgets.HTML("<span style='margin-left:1em'></span>"),
            self.save_btn
        ])
        goto_row = widgets.HBox([self.row_field, self.col_field, self.go_btn, self.size_slider])
        self.container = widgets.VBox([self.info_label, self.image_widget, self.status_label, buttons_row1, goto_row])

        # Events
        self.prev_btn.on_click(lambda b: self.prev_image())
        self.next_btn.on_click(lambda b: self.next_image())
        self.damaged_btn.on_click(lambda b: self.mark_damaged())
        self.undamaged_btn.on_click(lambda b: self.mark_undamaged())
        self.save_btn.on_click(lambda b: self.save_checkpoint())
        self.go_btn.on_click(lambda b: self.goto_rc(self.row_field.value, self.col_field.value))
        self.size_slider.observe(self._on_size_change, names='value')

        # Initial render
        self._render()

    def rc_from_idx(self, idx: int) -> Tuple[int, int]:
        r = idx // self.C
        c = idx % self.C

        return r, c

    def idx_from_rc(self, r: int, c: int) -> int:
        return r * self.C + c

    def _current_image(self) -> np.ndarray:
        r, c = self.rc_from_idx(self.idx)

        return self.array[r, c]  # (H, W, C)

    def mark_damaged(self):
        r, c = self.rc_from_idx(self.idx)
        if (r, c) not in self.damaged:
            self.damaged.add((r, c))
            self.json_store.save_damaged_set(self.file_id, self.damaged)

        self.next_image()

    def mark_undamaged(self):
        r, c = self.rc_from_idx(self.idx)
        if (r, c) in self.damaged:
            self.damaged.discard((r, c))
            self.json_store.save_damaged_set(self.file_id, self.damaged)

        self.next_image()

    def next_image(self):
        if self.idx < self.total - 1:
            self.idx += 1
        else:
            self.idx = 0

        self._render()

    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
        else:
            self.idx = self.total - 1

        self._render()

    def goto_rc(self, r: int, c: int):
        r = int(max(0, min(self.R - 1, r)))
        c = int(max(0, min(self.C - 1, c)))
        self.idx = self.idx_from_rc(r, c)
        self._render()

    def save_checkpoint(self):
        r, c = self.rc_from_idx(self.idx)
        self.json_store.save_checkpoint(self.file_id, r, c, self.idx)

        # Update status with a small note
        self.status_label.value = self.status_label.value + " &nbsp; | &nbsp; <b>Checkpoint saved.</b>"

    def _on_size_change(self, change):
        if change and change.get('name') == 'value':
            self.max_display_px = int(change['new'])
            self._render()

    def _render(self):
        r, c = self.rc_from_idx(self.idx)

        # Info & status
        dmg = (r, c) in self.damaged
        progress = f"{self.idx + 1}/{self.total}"
        self.info_label.value = (
            f"<b>File:</b> {self.file_id} &nbsp; | &nbsp; "
            f"<b>Shape:</b> ({self.R}, {self.C}, {self.H}, {self.W}, {self.Ch})"
        )
        self.status_label.value = (
            f"<b>Row:</b> {r} &nbsp; <b>Col:</b> {c} &nbsp; | &nbsp; "
            f"<b>Status:</b> {'<span style=\"color:#c00\">Damaged</span>' if dmg else 'Undamaged'} &nbsp; | &nbsp; "
            f"<b>Damaged count:</b> {len(self.damaged)} &nbsp; | &nbsp; "
            f"<b>Progress:</b> {progress}"
        )

        # Update row/col fields
        self.row_field.value = r
        self.col_field.value = c

        # Render image
        try:
            pil = array_to_pil(self._current_image())
        except Exception as e:
            pil = Image.new('RGB', (640, 480), color=(30, 30, 30))
            try:
                from PIL import ImageDraw
                d = ImageDraw.Draw(pil)
                d.text((10, 10), f"Error displaying image at (r={r}, c={c}): {e}", fill=(200, 200, 200))
            except Exception:
                pass

        pil_fit = fit_image_to_max_side(pil, self.max_display_px)
        self.image_widget.value = pil_to_png_bytes(pil_fit)
        self.image_widget.layout = widgets.Layout(width=f"{pil_fit.size[0]}px", height=f"{pil_fit.size[1]}px")

        # Auto-save checkpoint
        self.json_store.save_checkpoint(self.file_id, r, c, self.idx)

    def ui(self) -> widgets.VBox:
        return self.container


def load_array(input_path: str, key: Optional[str] = None, no_mmap: bool = False) -> np.ndarray:
    input_path = os.path.expanduser(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext == '.npy':
        mmap_mode = None if no_mmap else 'r'
        arr = np.load(input_path, mmap_mode=mmap_mode)
        arr = arr[..., ::-1].copy()

        return arr
    elif ext == '.npz':
        if key is None:
            raise ValueError("For .npz files you must provide 'key' for the array inside the archive.")
        with np.load(input_path) as z:
            if key not in z:
                raise KeyError(f"Key '{key}' not found in {input_path}. Available: {list(z.keys())}")
            
            return z[key]
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .npy (recommended) or .npz with key.")


def launch_notebook_labeler(
    input_path: str,
    json_path: str = './calibrations/damaged_segments.json',
    key: Optional[str] = None,
    start_row: Optional[int] = None,
    start_col: Optional[int] = None,
    no_mmap: bool = False,
    max_display_px: int = 768
):
    """
    Launch the damaged segment labeler in a Jupyter notebook.
    
    Parameters
    ----------
    input_path: str
        Path to the input .npy or .npz file containing the 5D array (H_seg, V_seg, height, width, num_channels).
    json_path: str
        Path to the JSON file to store damaged segment labels.
    key: str, optional
        If input_path is a .npz file, the key for the array inside the archive.
    start_row: int, optional
        Row index to start labeling from. If not provided, uses checkpoint or starts from 0.
    start_col: int, optional
        Column index to start labeling from. If not provided, uses checkpoint or starts from 0.
    no_mmap: bool, optional
        If True, disables memory mapping when loading .npy files.
    max_display_px: int, optional
        Maximum size of the displayed image's longest side in pixels.
    """
    arr = load_array(input_path, key=key, no_mmap=no_mmap)
    labeler = NotebookDamagedLabeler(
        array=arr,
        file_id=input_path,
        json_path=json_path,
        start_row=start_row,
        start_col=start_col,
        max_display_px=max_display_px
    )
    ui = labeler.ui()

    # Attach the instance for programmatic access if needed
    setattr(ui, '_labeler', labeler)

    return ui


def launch_from_array(
    array: np.ndarray,
    file_id: str,
    json_path: str = './calibrations/damaged_segments.json',
    start_row: Optional[int] = None,
    start_col: Optional[int] = None,
    max_display_px: int = 768
):
    labeler = NotebookDamagedLabeler(
        array=array,
        file_id=file_id,
        json_path=json_path,
        start_row=start_row,
        start_col=start_col,
        max_display_px=max_display_px
    )
    ui = labeler.ui()
    setattr(ui, '_labeler', labeler)

    return ui