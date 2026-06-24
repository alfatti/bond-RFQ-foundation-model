#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Production point-in-time embedding extraction (H200 box).

Loads a trained FM checkpoint, tokenizes RFQs with the cuDF fast path, and
extracts point-in-time embeddings with the GPU-gather forward (bf16). Emits
embeddings.parquet keyed by (client_id, time_rank) for the lift harness.

Only the clients that have rows in eval cells (B/D) plus their prior context
are extracted, so we don't embed RFQs no metric will use. Pass --all-cells to
also extract Cell A (e.g. to train the downstream head on embeddings).

Usage:
  python scripts/extract_embeddings_gpu.py \
      --rfq-dir data/igdesk \
      --corpus-dir data/rfq_corpus \
      --checkpoint models/bond-rfq-fm/checkpoints/<step> \
      --out data/rfq_corpus/embeddings.parquet \
      --score-block 256 --ctx-carry 184 --batch-windows 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import RFQTokenizer  # noqa: E402
from src.tokenizer.rfq_tokenizer import FIELD_ORDER  # noqa: E402
from src.extract_embeddings import (  # noqa: E402
    PointInTimeExtractor,
    ExtractionConfig,
    make_gpu_gather,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfq-dir", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-block", type=int, default=256)
    ap.add_argument("--ctx-carry", type=int, default=184)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--all-cells", action="store_true",
                    help="also extract Cell A (for training the embedding head)")
    ap.add_argument("--gpu-tokenizer", action="store_true",
                    help="use cuDF fast-path tokenizer")
    args = ap.parse_args()

    import torch
    from transformers import LlamaForCausalLM

    # ---- load RFQs + manifest -------------------------------------------
    import polars as pl
    files = sorted(Path(args.rfq_dir).glob("week=*/*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files]).to_pandas()
    df = df.sort_values(["timestamp", "rfq_id"], kind="stable").reset_index(drop=True)
    df["time_rank"] = np.arange(len(df))

    manifest = pd.read_parquet(Path(args.corpus_dir) / "manifest.parquet")
    cells = ["B", "D"] + (["A"] if args.all_cells else [])
    eval_clients = pd.Index(
        manifest[manifest["cell"].isin(cells)]["client_id"].unique()
    )
    sub = df[df["client_id"].isin(eval_clients)].copy()
    print(f"clients: {len(eval_clients):,} | RFQ rows incl. context: {len(sub):,}")

    # ---- tokenize -------------------------------------------------------
    state = json.load(open(Path(args.corpus_dir) / "rfq_vocab.json"))
    tok = RFQTokenizer.from_state(state)
    if args.gpu_tokenizer:
        import cudf
        from src.tokenizer.rfq_tokenizer_gpu import RFQTokenizerGPU
        gtok = RFQTokenizerGPU(state)
        words = gtok.tokenize_rfqs(cudf.from_pandas(sub)).get()
    else:
        words = tok.tokenize_rfqs(sub)

    # ---- model + GPU gather --------------------------------------------
    model = LlamaForCausalLM.from_pretrained(args.checkpoint).eval()
    D = model.config.hidden_size
    gather = make_gpu_gather(model, device="cuda", dtype="bfloat16")

    ext = PointInTimeExtractor(
        model_fn=None,  # oracle path unused in production
        n_fields=len(FIELD_ORDER), pad_id=tok.pad_id, bos_id=tok.bos_id,
        sep_id=tok.sep_id, eos_id=tok.eos_id, embed_dim=D,
        cfg=ExtractionConfig(
            score_block=args.score_block, ctx_carry=args.ctx_carry,
            batch_windows=args.batch_windows, max_seq_len=args.max_seq_len,
        ),
        gpu_gather=gather,
    )

    res = ext.extract(sub["client_id"].to_numpy(), words, sub["time_rank"].to_numpy())
    print(f"extracted {len(res['time_rank']):,} point-in-time embeddings, dim={D}")

    emb = pd.DataFrame(res["embedding"], columns=[f"e{i}" for i in range(D)])
    emb.insert(0, "history_len", res["history_len"])
    emb.insert(0, "time_rank", res["time_rank"])
    emb.insert(0, "client_id", res["client_id"])
    emb.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
