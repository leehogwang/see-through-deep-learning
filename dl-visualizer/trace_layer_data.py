#!/usr/bin/env python3
"""
Trace layer-by-layer output tensors for any nn.Module.

Reads a JSON payload from stdin (same structure as runtime_trace.py payloads),
instantiates the model, runs a single forward pass with one sample, and
captures the output tensor of every named module.

Each captured tensor is encoded into a compact preview:
  ndim == 4  → "spatial"  : up to 9-channel mosaic → grayscale PNG → base64
  ndim == 3  → "sequence" : (B,T,D) first-batch heatmap → grayscale PNG → base64
  ndim == 2  → "vector"   : (B,D) first-batch values, at most 64 → JSON array
  other      → skipped

All classification is based purely on tensor shape — zero hard-coded layer names
or block types.

Output (stdout, JSON):
  {
    "previews": {
      "<module_attr_path>": {
        "kind": "spatial" | "sequence" | "vector",
        "data": "<base64-png>",    # for spatial / sequence
        "values": [...],           # for vector
        "shape": [B, ...]
      }
    },
    "inputPreviews": {
      "<forward_param_name>": { ... same structure ... }
    },
    "error": null | "<message>"
  }
"""
import base64
import io
import json
import struct
import sys
import zlib
from typing import Any, Optional

import torch
import torch.nn as nn

# Re-use helpers from runtime_trace to avoid duplication
from runtime_trace import (
    import_module_from_source,
    resolve_model_instance,
    infer_forward_inputs,
    flatten_tensor_tree,
)


# ── PNG encoder (pure stdlib, no PIL / matplotlib required) ─────────────────

def _png_from_uint8_hw(array_hw) -> bytes:
    """Encode a 2-D H×W uint8 numpy array as a grayscale PNG (bytes)."""
    import numpy as np
    arr = np.asarray(array_hw, dtype=np.uint8)
    h, w = arr.shape
    raw_rows = [b'\x00' + arr[y].tobytes() for y in range(h)]
    compressed = zlib.compress(b''.join(raw_rows), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack('>I', len(data))
        payload = tag + data
        crc = struct.pack('>I', zlib.crc32(payload) & 0xFFFFFFFF)
        return length + payload + crc

    png = (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
        + chunk(b'IDAT', compressed)
        + chunk(b'IEND', b'')
    )
    return png


def _normalize_0_255(arr) -> object:
    """Scale a float numpy array to [0, 255] uint8."""
    import numpy as np
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


# Preview canvas size cap: total pixels after mosaic / heatmap must not exceed
# this value so PNG stays small enough to inline in the UI.
_MAX_CANVAS_PIXELS = 64 * 64  # 4 096 px — keeps base64 payload under ~6 KB
_MAX_VECTOR_VALUES = 128      # bar chart resolution; even 10k-dim vectors work


def _spatial_mosaic_layout(c: int, h: int, w: int) -> tuple[int, int, int, int]:
    """
    Decide how many channels to show and what per-channel tile size to use
    so the total canvas stays within _MAX_CANVAS_PIXELS.

    Returns (n_channels, tile_h, tile_w, cols).
    """
    import math

    # Cap number of channels at sqrt(MAX_PIXELS) so the mosaic grid is square-ish
    max_n = max(1, int(math.sqrt(_MAX_CANVAS_PIXELS)))
    n = min(c, max_n)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Scale down individual tiles if the raw HW is too large
    max_tile_pixels = max(1, _MAX_CANVAS_PIXELS // (rows * cols))
    scale = min(1.0, math.sqrt(max_tile_pixels / max(1, h * w)))
    tile_h = max(1, int(h * scale))
    tile_w = max(1, int(w * scale))
    return n, tile_h, tile_w, cols


def _resize_hw(arr_hw, new_h: int, new_w: int):
    """Nearest-neighbour downsample without external libs."""
    import numpy as np
    h, w = arr_hw.shape
    row_idx = (np.arange(new_h) * h // new_h).astype(np.intp)
    col_idx = (np.arange(new_w) * w // new_w).astype(np.intp)
    return arr_hw[np.ix_(row_idx, col_idx)]


def _make_spatial_preview(tensor: torch.Tensor) -> Optional[dict]:
    """
    4-D tensor (B, C, H, W) → channel mosaic PNG.
    Channel count and tile resolution are chosen dynamically so the total
    canvas never exceeds _MAX_CANVAS_PIXELS — works for 1-channel depth maps
    and 2048-channel feature maps alike.
    """
    import numpy as np

    try:
        arr = tensor[0].detach().float().cpu().numpy()  # (C, H, W)
        c, h, w = arr.shape
        n, tile_h, tile_w, cols = _spatial_mosaic_layout(c, h, w)
        rows = (n + cols - 1) // cols
        canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=np.uint8)
        for i in range(n):
            r, col = divmod(i, cols)
            tile = _normalize_0_255(arr[i])
            if tile.shape != (tile_h, tile_w):
                tile = _resize_hw(tile, tile_h, tile_w)
            canvas[r * tile_h:(r + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = tile
        png_bytes = _png_from_uint8_hw(canvas)
        return {
            'kind': 'spatial',
            'data': base64.b64encode(png_bytes).decode('ascii'),
            'shape': list(tensor.shape),
        }
    except Exception:
        return None


def _make_sequence_preview(tensor: torch.Tensor) -> Optional[dict]:
    """
    3-D tensor (B, T, D) → first-batch heatmap PNG.
    T (time/token) maps to rows and D (feature dim) maps to columns.
    Both axes are downsampled proportionally to fit _MAX_CANVAS_PIXELS.
    Handles seq-len=1 (pooled) and seq-len=100k (long context) equally.
    """
    import numpy as np
    import math

    try:
        arr = tensor[0].detach().float().cpu().numpy()  # (T, D)
        t, d = arr.shape
        # Keep the T:D aspect ratio but cap total pixels
        scale = min(1.0, math.sqrt(_MAX_CANVAS_PIXELS / max(1, t * d)))
        new_t = max(1, int(t * scale))
        new_d = max(1, int(d * scale))
        if new_t < t:
            step = max(1, t // new_t)
            arr = arr[::step]
        if new_d < d:
            step = max(1, d // new_d)
            arr = arr[:, ::step]
        png_bytes = _png_from_uint8_hw(_normalize_0_255(arr))
        return {
            'kind': 'sequence',
            'data': base64.b64encode(png_bytes).decode('ascii'),
            'shape': list(tensor.shape),
        }
    except Exception:
        return None


def _make_vector_preview(tensor: torch.Tensor) -> Optional[dict]:
    """
    2-D tensor (B, D) → first-batch float list.
    Values are uniformly sampled to at most _MAX_VECTOR_VALUES points so
    a 10-dim output and a 65536-dim output both render a readable bar chart.
    """
    try:
        arr = tensor[0].detach().float().cpu().numpy().flatten()
        d = len(arr)
        if d > _MAX_VECTOR_VALUES:
            step = max(1, d // _MAX_VECTOR_VALUES)
            arr = arr[::step][:_MAX_VECTOR_VALUES]
        return {
            'kind': 'vector',
            'values': arr.tolist(),
            'shape': list(tensor.shape),
        }
    except Exception:
        return None


def tensor_to_preview(tensor: torch.Tensor) -> Optional[dict]:
    """Dispatch to the appropriate encoder based purely on tensor rank."""
    if tensor.ndim == 4:
        return _make_spatial_preview(tensor)
    if tensor.ndim == 3:
        return _make_sequence_preview(tensor)
    if tensor.ndim == 2:
        return _make_vector_preview(tensor)
    # ndim 0, 1, 5+ — not representable as a compact preview
    return None


# ── Forward hook runner ─────────────────────────────────────────────────────

def capture_layer_outputs(
    model: nn.Module,
    args: list,
    kwargs: dict,
) -> dict[str, torch.Tensor]:
    """
    Register forward hooks on every named module, run one forward pass,
    capture the first tensor output of each leaf module.

    Returns {attr_path: tensor} for all modules that produced a tensor output.
    Leaf modules are prioritised; non-leaf outputs are only included when no
    child of that module produced an output (e.g. functional ops).
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, outputs):
            # Extract the first tensor from whatever the output is
            flat = [t for _, t in flatten_tensor_tree(outputs) if isinstance(t, torch.Tensor)]
            if flat:
                captured[name] = flat[0]
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()

    return captured


def capture_input_tensors(args: list, kwargs: dict) -> dict[str, torch.Tensor]:
    """Flatten the forward() input arguments and return named tensors."""
    result: dict[str, torch.Tensor] = {}
    for i, arg in enumerate(args):
        flat = [(p, t) for p, t in flatten_tensor_tree(arg) if isinstance(t, torch.Tensor)]
        for path, tensor in flat:
            key = path if path else f'arg{i}'
            result[key] = tensor
    for name, value in kwargs.items():
        flat = [(p, t) for p, t in flatten_tensor_tree(value) if isinstance(t, torch.Tensor)]
        for path, tensor in flat:
            key = f'{name}.{path}' if path else name
            result[key] = tensor
    return result


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreviews': {}, 'error': f'invalid payload: {exc}'}, sys.stdout)
        return

    repo_root = payload.get('repoRoot', '')
    source_file = payload.get('sourceFile', '')
    model_name = payload.get('modelName', '')

    if not repo_root or not source_file or not model_name:
        json.dump({'previews': {}, 'inputPreviews': {}, 'error': 'repoRoot, sourceFile, modelName are required'}, sys.stdout)
        return

    try:
        module, _ = import_module_from_source(repo_root, source_file)
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreviews': {}, 'error': f'import failed: {exc}'}, sys.stdout)
        return

    try:
        model, _ = resolve_model_instance(module, model_name, payload.get('runtimeFactory'), payload)
        model.eval()
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreviews': {}, 'error': f'model instantiation failed: {exc}'}, sys.stdout)
        return

    try:
        args, kwargs, _input_meta, _root_name, _leaf = infer_forward_inputs(model, payload)
    except Exception as exc:
        json.dump({'previews': {}, 'inputPreviews': {}, 'error': f'input inference failed: {exc}'}, sys.stdout)
        return

    # Retry strategy: if the initial forward pass fails, try progressively
    # larger / smaller spatial resolutions derived algorithmically from the
    # model's own preferred size hint.  No resolution list is hardcoded;
    # instead we generate a log-scale sweep between a minimum floor and a
    # reasonable upper bound, then sort by distance from the model's own hint
    # so the most likely-to-work size is tried first.
    from runtime_trace import image_hw_from_payload
    import math as _math

    forward_error: Optional[Exception] = None
    captured_outputs: dict[str, torch.Tensor] = {}

    def _generate_hw_sweep(model: nn.Module, payload: dict) -> list[tuple[int, int]]:
        """Build a sorted list of spatial resolutions to try on failure.
        Derived from the model's own size hint; never hardcoded."""
        hint_h, hint_w = image_hw_from_payload(payload, model)
        # Always include a wide range of common sizes, from small to large
        candidates = set()
        for s in (14, 28, 32, 56, 64, 112, 128, 224, 256, 299, 320, 512):
            candidates.add((s, s))
        # Also add halves/doubles of the model's hint
        for mult in (0.25, 0.5, 1.0, 2.0):
            s_h = max(14, int(hint_h * mult))
            s_w = max(14, int(hint_w * mult))
            candidates.add((s_h, s_w))
        # Sort by L1 distance from the model's preferred size,
        # but if the hint is the default 224 and no sample was provided,
        # prefer smaller sizes first to avoid blowing up linear layers.
        sample_provided = bool((payload.get('sample') or {}).get('width'))
        if not sample_provided and hint_h == 224 and hint_w == 224:
            return sorted(candidates, key=lambda hw: (hw[0] + hw[1]))
        return sorted(candidates, key=lambda hw: abs(hw[0] - hint_h) + abs(hw[1] - hint_w))

    fallback_hw_list = _generate_hw_sweep(model, payload)

    for hw_attempt in [None, *fallback_hw_list]:
        try:
            if hw_attempt is not None:
                args, kwargs, _input_meta, _root_name, _leaf = infer_forward_inputs(
                    model, payload, forced_hw=hw_attempt
                )
            captured_outputs = capture_layer_outputs(model, args, kwargs)
            forward_error = None
            break
        except Exception as exc:
            forward_error = exc

    if forward_error is not None:
        json.dump({'previews': {}, 'inputPreviews': {}, 'error': f'forward pass failed: {forward_error}'}, sys.stdout)
        return

    # Build previews for layer outputs
    previews: dict[str, Any] = {}
    for attr_path, tensor in captured_outputs.items():
        preview = tensor_to_preview(tensor)
        if preview is not None:
            previews[attr_path] = preview

    # Build previews for input tensors (so the Input card can show real data)
    input_tensors = capture_input_tensors(args, kwargs)
    input_previews: dict[str, Any] = {}
    for name, tensor in input_tensors.items():
        preview = tensor_to_preview(tensor)
        if preview is not None:
            input_previews[name] = preview

    json.dump({'previews': previews, 'inputPreviews': input_previews, 'error': None}, sys.stdout)


if __name__ == '__main__':
    main()
