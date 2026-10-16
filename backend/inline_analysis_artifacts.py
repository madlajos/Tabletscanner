"""Disk-backed analytical results, without pickle or retained camera arrays."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_data(directory: Path, data: dict) -> None:
    """Keep arrays losslessly on disk; JSON describes only this owned artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    array_count = 0

    def encode(value):
        nonlocal array_count
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return encode(value.tolist())
            filename = f'array-{array_count}.npy'
            array_count += 1
            np.save(directory / filename, value, allow_pickle=False)
            return {'__inline_array__': filename}
        if isinstance(value, dict):
            # Branch execution caches are implementation details, often carrying
            # complete duplicate input stacks. They are never preview outputs.
            return {str(k): encode(v) for k, v in value.items()
                    if not str(k).startswith('_') or k in (
                        '_original_count', '_original_paths', '_inline_capture_input')}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    write_json(directory / 'data.json', encode(data))


def load_data(directory: Path) -> dict:
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'__inline_array__'}:
                filename = value['__inline_array__']
                if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.npy'):
                    raise ValueError('Invalid analysis array reference.')
                return np.load(directory / filename, allow_pickle=False, mmap_mode='r')
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(json.loads((directory / 'data.json').read_text(encoding='utf-8')))
