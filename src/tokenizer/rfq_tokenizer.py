# SPDX-License-Identifier: Apache-2.0
"""
RFQ tokenizer for the Bond-RFQ foundation model.

Forked from NVIDIA's transaction-foundation-model tokenizer design: the same
modular-field + global-vocab-with-offsets + corpus-line architecture, adapted
from card transactions to corporate-bond RFQs.

KEY DESIGN DECISIONS (see project strategy):
  * EXOGENOUS FIELDS ONLY. The embedding is a policy-independent client-demand
    prior, so the desk's own actions/results are NOT tokenized:
        excluded: our_quote, cover_price, status, client_id, prices.
  * Each RFQ -> a fixed ordered sequence of field tokens.
  * A client's RFQ history -> "<bos> rfq1 <sep> rfq2 <sep> ... <eos>".
  * client_id is deliberately NOT a token: the model must INFER client state
    from behaviour, not memorise an identity embedding (which cannot transfer
    to held-out clients).

This module is backend-agnostic at the field level (pure column->token-string
maps), so the cuDF production path and the pandas dev path share one code path
for vocab + corpus assembly. Only the dataframe engine differs.

Fields and their tokenizer treatment
------------------------------------
    client_type   FIXED   5   direct categorical
    client_tier   FIXED   3
    sector        FIXED   8
    mat_bucket    FIXED   5   instrument maturity bucket (exogenous)
    age_bucket    FIXED   3   on-the-run age bucket (exogenous)
    side          MAP     2   BUY / SELL  (client perspective)
    size_bucket   QUANTILE    data-driven; fine-grained (behaviourally rich)
    k_dealers     FIXED   clipped competition count
    mmpp_state    FIXED   4   latent liquidity regime the client acted in

`size` is the only continuous field kept, and it is quantile-binned (not the
fixed-threshold scheme TFM used for amount) because size is tier/type-linked
behaviour we want resolved finely.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Field configuration
# ---------------------------------------------------------------------------

# Fields whose token space is a fixed small integer range [0, n).
# name -> (prefix, n_values)
FIXED_FIELDS: Dict[str, Tuple[str, int]] = {
    "client_type": ("CT", 5),
    "client_tier": ("TIER", 3),
    "sector": ("SEC", 8),
    "mat_bucket": ("MAT", 5),
    "age_bucket": ("AGE", 3),
    "mmpp_state": ("REG", 4),
}

# k_dealers is fixed-range but we clip the tail explicitly.
K_DEALERS_PREFIX = "K"
K_DEALERS_MAX = 8  # clip k_dealers to [1, 8]

# side is a 2-value map.
SIDE_PREFIX = "SIDE"
SIDE_MAP = {"SELL": 0, "BUY": 1}

# size is quantile-binned at fit time.
SIZE_PREFIX = "SZ"
SIZE_DEFAULT_N_BINS = 32  # fine-grained: size carries tier/type signal

# Field emission order within a single RFQ "word".
FIELD_ORDER: List[str] = [
    "client_type",
    "client_tier",
    "sector",
    "mat_bucket",
    "age_bucket",
    "side",
    "size_bucket",
    "k_dealers",
    "mmpp_state",
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class RFQTokenizer:
    """Builds the global vocab and converts RFQ rows to token-id sequences.

    Backend-neutral: operates on numpy arrays / pandas Series. The cuDF path
    passes cudf Series in; the dev path passes pandas Series. Only `size`
    binning needs the values materialised, which both backends support.
    """

    def __init__(
        self,
        size_n_bins: int = SIZE_DEFAULT_N_BINS,
        k_dealers_max: int = K_DEALERS_MAX,
    ):
        self.size_n_bins = size_n_bins
        self.k_dealers_max = k_dealers_max

        # fitted state
        self.size_edges: Optional[np.ndarray] = None  # quantile bin edges
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.field_token_count: Dict[str, int] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Vocab construction
    # ------------------------------------------------------------------

    def _all_field_tokens(self) -> List[str]:
        """Enumerate every possible field token (deterministic ordering)."""
        toks: List[str] = []

        for name, (prefix, n) in FIXED_FIELDS.items():
            toks.extend(f"{prefix}_{i}" for i in range(n))

        # side
        toks.extend(f"{SIDE_PREFIX}_{i}" for i in range(len(SIDE_MAP)))

        # size buckets (n_bins)
        toks.extend(f"{SIZE_PREFIX}_{i}" for i in range(self.size_n_bins))

        # k_dealers 1..k_max
        toks.extend(f"{K_DEALERS_PREFIX}_{i}" for i in range(1, self.k_dealers_max + 1))

        return toks

    def fit(self, size_values: np.ndarray) -> "RFQTokenizer":
        """Fit data-driven pieces (size quantile edges) and build global vocab.

        Parameters
        ----------
        size_values : 1-D array of RFQ sizes (float) drawn from the TRAIN cells
            only (never test) to avoid leakage into the bin edges.
        """
        sv = np.asarray(size_values, dtype=np.float64)
        sv = sv[np.isfinite(sv)]
        # quantile edges; log-space is natural for lognormal size but quantile
        # binning is already scale-adaptive, so use raw quantiles on log to get
        # roughly equal-population buckets.
        qs = np.linspace(0.0, 1.0, self.size_n_bins + 1)
        edges = np.quantile(np.log1p(sv), qs)
        # dedupe edges (ties at the floor of $100k) to keep bins monotonic
        edges = np.unique(edges)
        self.size_edges = edges
        # actual number of bins after dedupe
        self.size_n_bins = len(edges) - 1

        # build global vocab: specials first, then fields in canonical order
        self.token_to_id = {}
        for t in SPECIAL_TOKENS:
            self.token_to_id[t] = len(self.token_to_id)
        for t in self._all_field_tokens():
            if t not in self.token_to_id:
                self.token_to_id[t] = len(self.token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.is_fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<sep>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    # ------------------------------------------------------------------
    # Per-field tokenization (vectorised, numpy)
    # ------------------------------------------------------------------

    def _size_bucket_ids(self, size: np.ndarray) -> np.ndarray:
        x = np.log1p(np.asarray(size, dtype=np.float64))
        # searchsorted into interior edges -> bucket in [0, n_bins-1]
        b = np.searchsorted(self.size_edges[1:-1], x, side="right")
        return np.clip(b, 0, self.size_n_bins - 1).astype(np.int64)

    def tokenize_rfqs(self, df) -> np.ndarray:
        """Map an RFQ dataframe to an (n_rows, n_fields) int64 token-id matrix.

        Accepts any object exposing __getitem__ returning array-likes
        (pandas.DataFrame or cudf.DataFrame). Values are pulled to numpy.
        """
        if not self.is_fitted:
            raise RuntimeError("RFQTokenizer.fit() must be called first")

        def col(name):
            c = df[name]
            return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

        n = len(df)
        t2i = self.token_to_id
        unk = self.unk_id

        field_id_cols: List[np.ndarray] = []
        for field in FIELD_ORDER:
            if field in FIXED_FIELDS:
                prefix, nval = FIXED_FIELDS[field]
                v = col(field).astype(np.int64)
                v = np.clip(v, 0, nval - 1)
                ids = np.fromiter(
                    (t2i.get(f"{prefix}_{x}", unk) for x in v),
                    count=n,
                    dtype=np.int64,
                )
            elif field == "side":
                raw = col("side")
                codes = np.array(
                    [SIDE_MAP.get(str(s), 0) for s in raw], dtype=np.int64
                )
                ids = np.fromiter(
                    (t2i.get(f"{SIDE_PREFIX}_{x}", unk) for x in codes),
                    count=n,
                    dtype=np.int64,
                )
            elif field == "size_bucket":
                b = self._size_bucket_ids(col("size"))
                ids = np.fromiter(
                    (t2i.get(f"{SIZE_PREFIX}_{x}", unk) for x in b),
                    count=n,
                    dtype=np.int64,
                )
            elif field == "k_dealers":
                v = np.clip(col("k_dealers").astype(np.int64), 1, self.k_dealers_max)
                ids = np.fromiter(
                    (t2i.get(f"{K_DEALERS_PREFIX}_{x}", unk) for x in v),
                    count=n,
                    dtype=np.int64,
                )
            else:
                raise KeyError(field)
            field_id_cols.append(ids)

        return np.stack(field_id_cols, axis=1)  # (n, n_fields)

    @property
    def n_fields(self) -> int:
        return len(FIELD_ORDER)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "size_n_bins": self.size_n_bins,
            "k_dealers_max": self.k_dealers_max,
            "size_edges": None if self.size_edges is None else self.size_edges.tolist(),
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_state(cls, state: dict) -> "RFQTokenizer":
        obj = cls(size_n_bins=state["size_n_bins"], k_dealers_max=state["k_dealers_max"])
        obj.size_edges = (
            None if state["size_edges"] is None else np.asarray(state["size_edges"])
        )
        obj.token_to_id = {k: int(v) for k, v in state["token_to_id"].items()}
        obj.id_to_token = {v: k for k, v in obj.token_to_id.items()}
        obj.is_fitted = obj.size_edges is not None
        return obj
