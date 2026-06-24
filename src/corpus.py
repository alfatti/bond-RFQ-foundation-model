# SPDX-License-Identifier: Apache-2.0
"""
Leakage-safe corpus construction for the Bond-RFQ foundation model.

Implements the 2x2 (time x client) partition agreed in the project strategy and
the three hard invariants that make every downstream metric trustworthy:

    INVARIANT 1  Pretrain ONLY on Cell A (seen clients x train weeks).
                 -> enforces train-only pretraining AND holdout-client novelty.
    INVARIANT 2  Every embedding pools a client's STRICTLY-PRIOR RFQs, read at
                 the position BEFORE the scored RFQ (handled in extraction, not
                 here, but the corpus is ordered to make it exact).
    INVARIANT 3  Cells B and D are scored but never trained on; Cell C is
                 context-only (prefix history for holdout clients), never
                 trained on.

Partition
---------
                 |  TRAIN weeks            |  TEST weeks (last `test_weeks`)
    SEEN  (1-f)  |  Cell A  pretrain+head  |  Cell B  HEADLINE eval
    HOLDOUT (f)  |  Cell C  context-only   |  Cell D  generalization eval

This file produces:
  * Cell A corpus lines for CLM pretraining (per-client windowed sequences).
  * A row-level manifest tagging every RFQ with its cell, client, and global
    time order, so the extraction step can build point-in-time embeddings and
    the eval step can pull the right rows.

Sequence format per window (matches NVIDIA pipeline):
    "<bos> rfq1 <sep> rfq2 <sep> ... <eos>"
where each rfqK is the space-joined field tokens for that RFQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PartitionConfig:
    test_weeks: int = 6          # last N ISO-week buckets are the test tail
    holdout_frac: float = 0.10   # fraction of clients fully held out (Cells C/D)
    min_history: int = 20        # drop clients with < this many TRAIN-week RFQs
    window_rfqs: int = 440       # RFQs per pretraining sequence (~4096 tokens)
    window_stride: int = 220     # 50% overlap for deep histories
    seed: int = 0


@dataclass
class CorpusArtifacts:
    pretrain_lines: List[str]               # Cell A windowed sequences
    manifest: pd.DataFrame                  # row-level: rfq order, cell, client
    cell_counts: Dict[str, int]
    n_seen_clients: int
    n_holdout_clients: int
    dropped_clients: int


# ---------------------------------------------------------------------------
# Partition assignment
# ---------------------------------------------------------------------------


def assign_week_buckets(timestamps: np.ndarray) -> np.ndarray:
    """Map timestamps -> integer week index (0-based, contiguous)."""
    ts = pd.to_datetime(timestamps)
    # days since min, integer-divided into 7-day buckets -> contiguous weeks
    day0 = ts.min().normalize()
    days = (ts.normalize() - day0).days
    return (days // 7).astype(np.int64)


def partition_clients(
    client_ids: np.ndarray, holdout_frac: float, seed: int
) -> Tuple[np.ndarray, set]:
    """Deterministically split unique clients into seen / holdout."""
    uniq = np.unique(client_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_holdout = int(round(len(uniq) * holdout_frac))
    holdout = set(perm[:n_holdout].tolist())
    return uniq, holdout


def build_manifest(
    df: pd.DataFrame, cfg: PartitionConfig
) -> Tuple[pd.DataFrame, set]:
    """Tag every RFQ row with (week_bucket, is_test, is_holdout, cell).

    Requires df columns: client_id, timestamp. Preserves original row index
    so token matrices (built in the same row order) align positionally.
    """
    m = pd.DataFrame(index=df.index)
    m["client_id"] = df["client_id"].to_numpy()
    wk = assign_week_buckets(df["timestamp"].to_numpy())
    m["week_bucket"] = wk

    max_wk = wk.max()
    test_start = max_wk - cfg.test_weeks + 1
    m["is_test"] = wk >= test_start

    _, holdout = partition_clients(m["client_id"].to_numpy(), cfg.holdout_frac, cfg.seed)
    m["is_holdout"] = m["client_id"].isin(holdout).to_numpy()

    # cell labels
    cell = np.where(
        ~m["is_holdout"] & ~m["is_test"], "A",
        np.where(
            ~m["is_holdout"] & m["is_test"], "B",
            np.where(m["is_holdout"] & ~m["is_test"], "C", "D"),
        ),
    )
    m["cell"] = cell

    # global time order (stable) for point-in-time embedding alignment
    order = np.argsort(df["timestamp"].to_numpy(), kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    m["time_rank"] = rank

    return m, holdout


# ---------------------------------------------------------------------------
# Min-history filter (TRAIN-week counts only, per Invariant 1 spirit)
# ---------------------------------------------------------------------------


def apply_min_history(manifest: pd.DataFrame, min_history: int) -> Tuple[pd.DataFrame, int]:
    """Drop clients whose TRAIN-week (non-test) RFQ count < min_history.

    Counts use train-week rows only so the filter reflects usable pretraining /
    context history, not test-period activity.
    """
    train_rows = manifest[~manifest["is_test"]]
    counts = train_rows.groupby("client_id").size()
    keep_clients = set(counts[counts >= min_history].index.tolist())
    before = manifest["client_id"].nunique()
    filtered = manifest[manifest["client_id"].isin(keep_clients)].copy()
    dropped = before - len(keep_clients)
    return filtered, dropped


# ---------------------------------------------------------------------------
# Corpus line assembly (Cell A only)
# ---------------------------------------------------------------------------


def _rfq_words(token_matrix: np.ndarray, id_to_token: Dict[int, str]) -> np.ndarray:
    """Convert (n, n_fields) id matrix -> array of n space-joined token strings."""
    # vectorised join via python list comp (cuDF path uses str.cat instead)
    lut = id_to_token
    words = [
        " ".join(lut[i] for i in row)
        for row in token_matrix
    ]
    return np.asarray(words, dtype=object)


def build_pretrain_corpus(
    manifest: pd.DataFrame,
    token_matrix: np.ndarray,
    id_to_token: Dict[int, str],
    cfg: PartitionConfig,
) -> List[str]:
    """Assemble Cell A windowed per-client sequences.

    token_matrix rows are positionally aligned with manifest rows (same source
    df row order). We select Cell A, order each client's RFQs by time, and emit
    overlapping windows of `window_rfqs`.
    """
    cellA = manifest[manifest["cell"] == "A"]
    # map manifest positional rows -> token_matrix rows
    # manifest index is the original df index; token_matrix is in df order, so
    # we need the positional location of each cellA row within the df.
    pos = manifest.index.get_indexer(cellA.index)  # positions into token_matrix
    words = _rfq_words(token_matrix[pos], id_to_token)

    work = pd.DataFrame(
        {
            "client_id": cellA["client_id"].to_numpy(),
            "time_rank": cellA["time_rank"].to_numpy(),
            "word": words,
        }
    )
    work.sort_values(["client_id", "time_rank"], inplace=True, kind="stable")

    lines: List[str] = []
    w, s = cfg.window_rfqs, cfg.window_stride
    for _, grp in work.groupby("client_id", sort=False):
        seq = grp["word"].tolist()
        n = len(seq)
        if n == 0:
            continue
        if n <= w:
            lines.append("<bos> " + " <sep> ".join(seq) + " <eos>")
        else:
            start = 0
            while start < n:
                chunk = seq[start : start + w]
                lines.append("<bos> " + " <sep> ".join(chunk) + " <eos>")
                if start + w >= n:
                    break
                start += s
    return lines


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def build_corpus(
    df: pd.DataFrame,
    token_matrix: np.ndarray,
    id_to_token: Dict[int, str],
    cfg: PartitionConfig,
) -> CorpusArtifacts:
    """Full pipeline: partition -> filter -> Cell A corpus + manifest."""
    manifest, holdout = build_manifest(df, cfg)
    manifest, dropped = apply_min_history(manifest, cfg.min_history)

    lines = build_pretrain_corpus(manifest, token_matrix, id_to_token, cfg)

    cell_counts = manifest["cell"].value_counts().to_dict()
    n_holdout = manifest[manifest["is_holdout"]]["client_id"].nunique()
    n_seen = manifest[~manifest["is_holdout"]]["client_id"].nunique()

    return CorpusArtifacts(
        pretrain_lines=lines,
        manifest=manifest,
        cell_counts=cell_counts,
        n_seen_clients=n_seen,
        n_holdout_clients=n_holdout,
        dropped_clients=dropped,
    )


# ---------------------------------------------------------------------------
# Packed token-ID corpus (GPU/production path — no string round-trip)
# ---------------------------------------------------------------------------


def build_pretrain_corpus_packed(
    manifest: pd.DataFrame,
    token_matrix: np.ndarray,
    special_ids: Dict[str, int],
    cfg: PartitionConfig,
    n_fields: int,
):
    """Assemble Cell A windowed sequences directly as packed token IDs.

    Equivalent token-for-token to build_pretrain_corpus() but emits integer ids
    (no string assembly, no text parse at load). Returns:
        tokens  : 1-D int32, all sequences concatenated
        offsets : 1-D int64, length n_seq+1, sequence k = tokens[off[k]:off[k+1]]

    Sequence layout per window: bos, w0(F ids), sep, w1(F ids), sep, ..., eos.
    """
    bos, sep, eos = special_ids["<bos>"], special_ids["<sep>"], special_ids["<eos>"]
    F = n_fields

    cellA = manifest[manifest["cell"] == "A"]
    pos = manifest.index.get_indexer(cellA.index)
    words = token_matrix[pos]  # (n_cellA, F) int

    work = pd.DataFrame(
        {"client_id": cellA["client_id"].to_numpy(),
         "time_rank": cellA["time_rank"].to_numpy(),
         "row": np.arange(len(cellA))}
    ).sort_values(["client_id", "time_rank"], kind="stable")

    w, s = cfg.window_rfqs, cfg.window_stride
    seqs: List[np.ndarray] = []
    for _, grp in work.groupby("client_id", sort=False):
        idx = grp["row"].to_numpy()
        n = len(idx)
        if n == 0:
            continue
        starts = [0] if n <= w else list(range(0, n, s))
        for st in starts:
            chunk_rows = idx[st: st + w]
            m = len(chunk_rows)
            wmat = words[chunk_rows]  # (m, F)
            seq_len = 1 + m * F + (m - 1) + 1
            seq = np.empty(seq_len, dtype=np.int32)
            seq[0] = bos
            p = 1
            for k in range(m):
                seq[p:p + F] = wmat[k]
                p += F
                if k < m - 1:
                    seq[p] = sep
                    p += 1
            seq[p] = eos
            seqs.append(seq)
            if st + w >= n:
                break

    if not seqs:
        return np.zeros(0, dtype=np.int32), np.zeros(1, dtype=np.int64)
    lengths = np.array([len(x) for x in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tokens = np.concatenate(seqs).astype(np.int32)
    return tokens, offsets
