# SPDX-License-Identifier: Apache-2.0
"""
GPU fast-path RFQ tokenizer (cuDF / cuPy).

Produces the SAME (n, n_fields) int64 token-id matrix as the numpy reference
``RFQTokenizer`` in rfq_tokenizer.py, but fully vectorised on-GPU: no Python
per-element dict lookups. Use on the H200 box; the numpy version remains the
correctness oracle (see scripts/parity_test.py).

Strategy
--------
Each field's token id is an affine function of a small integer code:
    fixed fields:  id = base_offset[field] + clip(code, 0, n-1)
    side:          id = base + side_code (SELL=0, BUY=1)
    size:          id = base + searchsorted(interior_edges, log1p(size), 'right')
    k_dealers:     id = base + (clip(k,1,kmax) - 1)
Because the per-field token block is contiguous in the global vocab (the numpy
fit() lays them out in canonical FIELD/enumeration order), the id is just
``offset + local_code`` with no lookup. We precompute each field's offset from
the saved vocab so the GPU path and numpy path agree by construction.

This keeps parity EXACT for everything except the size bin edges, which depend
only on fitted floats shared via the saved state -> identical bins.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .rfq_tokenizer import (
    FIELD_ORDER,
    FIXED_FIELDS,
    K_DEALERS_PREFIX,
    SIDE_MAP,
    SIDE_PREFIX,
    SIZE_PREFIX,
    RFQTokenizer,
)


class RFQTokenizerGPU:
    """cuDF/cuPy fast path. Construct from a fitted numpy RFQTokenizer state."""

    def __init__(self, state: dict):
        # rebuild the reference to inherit vocab + edges exactly
        self._ref = RFQTokenizer.from_state(state)
        self.token_to_id: Dict[str, int] = self._ref.token_to_id
        self.size_edges = self._ref.size_edges
        self.size_n_bins = self._ref.size_n_bins
        self.k_dealers_max = self._ref.k_dealers_max
        self.unk_id = self._ref.unk_id

        # precompute per-field base offsets (id of local code 0) from the vocab,
        # so id = offset + local_code with NO lookup.
        t2i = self.token_to_id
        self._fixed_offset = {
            f: t2i[f"{prefix}_0"] for f, (prefix, _) in FIXED_FIELDS.items()
        }
        self._side_offset = t2i[f"{SIDE_PREFIX}_0"]
        self._size_offset = t2i[f"{SIZE_PREFIX}_0"]
        # k_dealers local code is (k-1), so offset corresponds to K_1
        self._k_offset = t2i[f"{K_DEALERS_PREFIX}_1"]

    @property
    def n_fields(self) -> int:
        return len(FIELD_ORDER)

    @property
    def vocab_size(self) -> int:
        return self._ref.vocab_size

    @property
    def pad_id(self): return self._ref.pad_id
    @property
    def bos_id(self): return self._ref.bos_id
    @property
    def eos_id(self): return self._ref.eos_id
    @property
    def sep_id(self): return self._ref.sep_id

    # ------------------------------------------------------------------
    # GPU tokenization
    # ------------------------------------------------------------------

    def tokenize_rfqs(self, df) -> "np.ndarray":
        """Map a cudf.DataFrame to an (n, n_fields) int64 token-id matrix (cupy).

        Returns a cupy array on GPU; call .get() for numpy if needed. Accepts a
        cudf.DataFrame (production) or pandas (falls back via cupy.asarray).
        """
        import cupy as cp

        def col(name):
            c = df[name]
            # cudf Series -> cupy; pandas -> numpy -> cupy
            if hasattr(c, "values") and hasattr(c.values, "__cuda_array_interface__"):
                return cp.asarray(c.values)
            return cp.asarray(np.asarray(c))

        n = len(df)
        out = cp.empty((n, self.n_fields), dtype=cp.int64)

        for fi, field in enumerate(FIELD_ORDER):
            if field in FIXED_FIELDS:
                _, nval = FIXED_FIELDS[field]
                code = cp.clip(col(field).astype(cp.int64), 0, nval - 1)
                out[:, fi] = self._fixed_offset[field] + code
            elif field == "side":
                # side stored as string "BUY"/"SELL" in cudf; map via equality
                s = df["side"]
                if s.dtype == object or str(s.dtype) == "object" or s.dtype.name == "str":
                    is_buy = cp.asarray((s == "BUY").values)
                    code = is_buy.astype(cp.int64)  # BUY=1, SELL=0
                else:
                    code = cp.clip(col("side").astype(cp.int64), 0, 1)
                out[:, fi] = self._side_offset + code
            elif field == "size_bucket":
                x = cp.log1p(col("size").astype(cp.float64))
                interior = cp.asarray(self.size_edges[1:-1])
                b = cp.searchsorted(interior, x, side="right")
                b = cp.clip(b, 0, self.size_n_bins - 1)
                out[:, fi] = self._size_offset + b
            elif field == "k_dealers":
                k = cp.clip(col("k_dealers").astype(cp.int64), 1, self.k_dealers_max)
                out[:, fi] = self._k_offset + (k - 1)
            else:
                raise KeyError(field)

        return out
