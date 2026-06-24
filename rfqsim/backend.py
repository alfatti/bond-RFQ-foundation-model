"""GPU/CPU backend abstraction.

All hot-path array code uses `xp` (CuPy on GPU, NumPy on CPU). Cold-path
(universe construction, MMPP chain, alias-table builds) stays on NumPy:
those are tiny and sequential, and shipping them to GPU would only add
transfer latency.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import cupy as cp  # noqa: F401

    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def gpu_available() -> bool:
    if not _HAS_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_xp(use_gpu: bool):
    """Return the array module for the hot path."""
    if use_gpu and gpu_available():
        return cp
    return np


def to_np(a):
    """Bring an array back to host memory (no-op on NumPy)."""
    if _HAS_CUPY and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def set_device(device_id: int) -> None:
    if gpu_available():
        cp.cuda.Device(device_id).use()
        # Async memory pool keeps allocations cheap across week-chunks.
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)


def pinned_rng(xp, seed: int):
    """Per-worker RNG. CuPy and NumPy share the Generator API surface we use."""
    if xp is np:
        return np.random.default_rng(seed)
    return cp.random.default_rng(seed)


def device_count() -> int:
    if not gpu_available():
        return 0
    return cp.cuda.runtime.getDeviceCount()


def worker_env_device() -> int:
    return int(os.environ.get("RFQSIM_DEVICE", "0"))
