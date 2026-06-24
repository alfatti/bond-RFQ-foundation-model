# SPDX-License-Identifier: Apache-2.0
"""
Point-in-time embedding extraction for the Bond-RFQ foundation model.

This is the leakage-critical component (Invariant 2). To score RFQ r_j of a
client, the embedding must summarize ONLY that client's RFQs strictly before j.
A causal decoder gives every prefix's hidden state in a single forward pass, so
we run one pass per client sequence and harvest, for each RFQ r_j, the hidden
state at the <sep> token that immediately precedes w_j (the field-word of r_j).
That state has causally attended to w_0..w_{j-1} and nothing of w_j.

Token layout for a sequence "<bos> w_0 <sep> w_1 <sep> ... <eos>" with F fields:
    word j field tokens occupy positions  [1 + j*(F+1),  1 + j*(F+1) + F)
    the <sep> immediately BEFORE word j is at position  j*(F+1)   for j>=1
    for j == 0 there is no prior history -> use the <bos> state at position 0
      and flag history_len == 0 so the eval can include/exclude.

Windowing for extraction
------------------------
Pretraining used overlapping windows. For extraction we must score each RFQ
exactly once from the longest available causal prefix, so we TILE each client's
history with NON-overlapping windows of `score_block` RFQs, but PREPEND a
context carry of the previous `ctx_carry` RFQs (encoded as warm-up, not scored)
so prefixes are not truncated at block boundaries.

    window k tokens:  <bos> [carry RFQs] [score RFQs] <eos>
    only the score RFQs' preceding-<sep> states are emitted.

Output: a DataFrame-ready set of arrays
    client_id, time_rank, history_len, embedding[d]
keyed so the eval joins to manifest rows by (client_id, time_rank).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExtractionConfig:
    score_block: int = 256       # RFQs scored per window
    ctx_carry: int = 184         # warm-up RFQs prepended (not scored)
    # score_block + ctx_carry should keep total tokens within model context.
    # (256+184)=440 RFQs * (F+1) + 2 ~ 4096 tokens for F=9.
    batch_windows: int = 16      # sequences per forward batch
    max_seq_len: int = 4096


class PointInTimeExtractor:
    """Backend-neutral driver around a HF causal LM forward pass.

    The model wrapper must expose:
        forward_hidden(input_ids_2d_int64, pad_id) -> last_layer_hidden (B,T,D)
    so the same driver works with a CPU HF model (validation) or a GPU-batched
    implementation (production).
    """

    def __init__(self, model_fn, n_fields: int, pad_id: int, bos_id: int,
                 sep_id: int, eos_id: int, embed_dim: int,
                 cfg: Optional[ExtractionConfig] = None, gpu_gather=None):
        self.model_fn = model_fn          # callable(ids2d, pad_id)->(B,T,D) np (oracle path)
        self._gpu_gather = gpu_gather     # optional callable(ids2d,pad,rows,poss)->(K,D) np
        self.F = n_fields
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.sep_id = sep_id
        self.eos_id = eos_id
        self.embed_dim = embed_dim
        self.cfg = cfg or ExtractionConfig()

    # ------------------------------------------------------------------
    # Build the (windowed) token sequences + scoring index maps
    # ------------------------------------------------------------------

    def _client_windows(
        self, words_ids: np.ndarray, time_ranks: np.ndarray
    ) -> List[Tuple[np.ndarray, List[Tuple[int, int, int]]]]:
        """Tile one client's history into context-carried scoring windows.

        Parameters
        ----------
        words_ids : (n_rfqs, F) int64 field-token ids, time-ordered.
        time_ranks : (n_rfqs,) int64 global time rank per RFQ (for join keys).

        Returns a list of (token_seq_1d, score_map) where score_map is a list of
        (token_position_of_preceding_sep, time_rank, history_len) for each
        SCORED rfq in this window.
        """
        F = self.F
        n = len(words_ids)
        block, carry = self.cfg.score_block, self.cfg.ctx_carry
        windows = []

        start = 0
        while start < n:
            score_lo = start
            score_hi = min(start + block, n)
            ctx_lo = max(0, score_lo - carry)   # warm-up region [ctx_lo, score_lo)

            seq_rfqs = words_ids[ctx_lo:score_hi]      # (m, F)
            m = len(seq_rfqs)
            # flat token sequence: <bos> w_0 <sep> w_1 <sep> ... w_{m-1} <eos>
            # length = 1 (bos) + m*F (words) + (m-1) (seps) + 1 (eos)
            seq_len = 1 + m * F + (m - 1) + 1
            seq = np.empty(seq_len, dtype=np.int64)
            seq[0] = self.bos_id
            p = 1
            for k in range(m):
                seq[p:p + F] = seq_rfqs[k]
                p += F
                if k < m - 1:
                    seq[p] = self.sep_id
                    p += 1
            seq[p] = self.eos_id  # final token (after last word, no trailing sep)
            assert p == seq_len - 1, (p, seq_len)

            # scoring map: for each rfq in [score_lo, score_hi), find its
            # in-window index and the preceding-<sep> token position.
            score_map: List[Tuple[int, int, int]] = []
            for j_global in range(score_lo, score_hi):
                k = j_global - ctx_lo                  # in-window word index
                hist_len = j_global                    # # of strictly-prior RFQs (global)
                if k == 0:
                    pos = 0                             # <bos> (only when no carry, i.e. global r_0)
                else:
                    pos = k * (F + 1)                  # the <sep> before word k
                    # NOTE: word k starts at 1 + k*(F+1); the sep is at k*(F+1).
                    # For k>=1 this indexes the sep that ends word k-1's group.
                score_map.append((pos, int(time_ranks[j_global]), int(hist_len)))

            windows.append((seq, score_map))
            if score_hi >= n:
                break
            start = score_hi  # non-overlapping scoring blocks
        return windows

    # ------------------------------------------------------------------
    # Run extraction over many clients
    # ------------------------------------------------------------------

    def extract(
        self,
        client_ids: np.ndarray,
        words_ids: np.ndarray,
        time_ranks: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract point-in-time embeddings for all given RFQs.

        Inputs are row-aligned arrays over the RFQs to be embedded (typically
        all of a cell's clients PLUS their prior-context rows). Rows need not be
        sorted; we group + sort per client internally.

        Returns dict of arrays: client_id, time_rank, history_len, embedding(d).
        """
        F = self.F
        order = np.lexsort((time_ranks, client_ids))  # sort by client, then time
        cid = client_ids[order]
        wid = words_ids[order]
        tr = time_ranks[order]

        # group boundaries
        uniq, starts = np.unique(cid, return_index=True)
        starts = list(starts) + [len(cid)]

        # accumulate windows across clients, then batch through the model
        all_seqs: List[np.ndarray] = []
        # per-window list of (row_in_batch placeholder, score_map)
        seq_score_maps: List[List[Tuple[int, int, int]]] = []
        seq_client: List[int] = []

        for gi in range(len(uniq)):
            lo, hi = starts[gi], starts[gi + 1]
            c_words = wid[lo:hi]
            c_tr = tr[lo:hi]
            for seq, smap in self._client_windows(c_words, c_tr):
                all_seqs.append(seq)
                seq_score_maps.append(smap)
                seq_client.append(int(uniq[gi]))

        # batch forward, harvesting needed positions
        out_cid: List[int] = []
        out_tr: List[int] = []
        out_hist: List[int] = []
        out_emb: List[np.ndarray] = []

        B = self.cfg.batch_windows
        maxT = self.cfg.max_seq_len
        for b0 in range(0, len(all_seqs), B):
            chunk = all_seqs[b0:b0 + B]
            T = min(maxT, max(len(s) for s in chunk))
            ids = np.full((len(chunk), T), self.pad_id, dtype=np.int64)
            for r, s in enumerate(chunk):
                L = min(len(s), T)
                ids[r, :L] = s[:L]

            # collect (row, pos) gather indices for this batch
            rows, poss, metas = [], [], []
            for r in range(len(chunk)):
                cl = seq_client[b0 + r]
                for (pos, trank, hlen) in seq_score_maps[b0 + r]:
                    if pos >= T:
                        continue
                    rows.append(r); poss.append(pos); metas.append((cl, trank, hlen))

            if self._gpu_gather is not None:
                # on-GPU gather: copy only (n_scored, D) back to host
                embs = self._gpu_gather(ids, self.pad_id,
                                        np.asarray(rows, dtype=np.int64),
                                        np.asarray(poss, dtype=np.int64))
            else:
                hidden = np.asarray(self.model_fn(ids, self.pad_id))  # (B,T,D)
                embs = hidden[np.asarray(rows), np.asarray(poss), :]

            for i, (cl, trank, hlen) in enumerate(metas):
                out_cid.append(cl); out_tr.append(trank); out_hist.append(hlen)
                out_emb.append(embs[i])

        return {
            "client_id": np.asarray(out_cid, dtype=np.int64),
            "time_rank": np.asarray(out_tr, dtype=np.int64),
            "history_len": np.asarray(out_hist, dtype=np.int64),
            "embedding": np.asarray(out_emb, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# GPU fast-path factory
# ---------------------------------------------------------------------------


def make_gpu_gather(model, device="cuda", dtype="bfloat16"):
    """Build an on-GPU gather closure for a torch HF causal LM.

    Returns a callable(ids2d_np, pad_id, rows_np, poss_np) -> (K, D) float32 np
    that runs the forward in bf16 autocast and gathers ONLY the requested
    (row, position) hidden states on-GPU, copying back just (K, D).

    This is the production path; the numpy ``model_fn`` remains the oracle for
    parity testing (scripts/parity_test.py asserts the two match).
    """
    import torch

    torch_dtype = getattr(torch, dtype)
    model = model.to(device).eval()

    @torch.no_grad()
    def gather(ids2d, pad_id, rows, poss):
        ids = torch.from_numpy(ids2d).to(device, non_blocking=True)
        am = (ids != pad_id).long()
        with torch.autocast(device_type="cuda", dtype=torch_dtype):
            out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
            h = out.hidden_states[-1]                      # (B, T, D)
        r = torch.from_numpy(rows).to(device)
        p = torch.from_numpy(poss).to(device)
        picked = h[r, p, :]                                # (K, D) on-GPU gather
        return picked.float().cpu().numpy()

    return gather
