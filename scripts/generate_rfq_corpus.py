#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate the Bond-RFQ pretraining corpus from simulated RFQ parquet.

Pipeline:
  1. Load RFQ parquet (all weeks).
  2. Compute week buckets; fit RFQTokenizer on TRAIN-week sizes only
     (size bin edges must not see test data -> Invariant against leakage).
  3. Tokenize all RFQs to a field-token-id matrix.
  4. Build the 2x2 (time x client) partition and Cell A windowed corpus.
  5. Write:
       <out>/cellA_train.txt   pretraining lines (90% of Cell A windows)
       <out>/cellA_val.txt     held-out validation lines (10%)
       <out>/rfq_vocab.json    saved tokenizer state (global vocab)
       <out>/manifest.parquet  row-level cell/client/time_rank tags
                               (consumed by the extraction + eval steps)

The manifest is the contract between pretraining and downstream eval: it lets
the extractor build point-in-time embeddings (Invariant 2) and lets the eval
pull Cell B / Cell D rows with the right labels.

Usage:
  python scripts/generate_rfq_corpus.py \
      --rfq-dir data/igdesk_dev \
      --out data/rfq_corpus \
      --test-weeks 6 --holdout-frac 0.10 --min-history 20 \
      --window 440 --stride 220 --size-bins 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo-root import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import RFQTokenizer  # noqa: E402
import src.corpus as corpus  # noqa: E402


def load_rfq_parquet(rfq_dir: str) -> pd.DataFrame:
    """Load all week partitions into a single (row-ordered) DataFrame."""
    try:
        import polars as pl

        files = sorted(Path(rfq_dir).glob("week=*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet under {rfq_dir}")
        df = pl.concat([pl.read_parquet(f) for f in files]).to_pandas()
    except ImportError:
        import pyarrow.dataset as ds

        df = ds.dataset(rfq_dir, partitioning="hive").to_table().to_pandas()
    # stable ordering by time then rfq_id so token matrix + manifest align
    df = df.sort_values(["timestamp", "rfq_id"], kind="stable").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfq-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-weeks", type=int, default=6)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--min-history", type=int, default=20)
    ap.add_argument("--window", type=int, default=440)
    ap.add_argument("--stride", type=int, default=220)
    ap.add_argument("--size-bins", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu",
                    help="gpu uses the cuDF/cuPy fast-path tokenizer (H200 box)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("[1/5] loading RFQ parquet ...")
    df = load_rfq_parquet(args.rfq_dir)
    print(f"      {len(df):,} RFQs | {df['client_id'].nunique()} clients")

    print("[2/5] fitting tokenizer on TRAIN-week sizes (leakage-safe) ...")
    wk = corpus.assign_week_buckets(df["timestamp"].to_numpy())
    test_start = wk.max() - args.test_weeks + 1
    train_mask = wk < test_start
    tok = RFQTokenizer(size_n_bins=args.size_bins)
    tok.fit(df.loc[train_mask, "size"].to_numpy())
    print(f"      vocab={tok.vocab_size} | size bins={tok.size_n_bins} "
          f"| fit rows={int(train_mask.sum()):,}")

    print(f"[3/5] tokenizing all RFQs (backend={args.backend}) ...")
    if args.backend == "gpu":
        import cudf
        from src.tokenizer.rfq_tokenizer_gpu import RFQTokenizerGPU
        gtok = RFQTokenizerGPU(tok.get_state())
        gdf = cudf.from_pandas(df)
        mat = gtok.tokenize_rfqs(gdf).get()  # cupy -> numpy
    else:
        mat = tok.tokenize_rfqs(df)

    print("[4/5] building 2x2 partition + Cell A corpus ...")
    pcfg = corpus.PartitionConfig(
        test_weeks=args.test_weeks,
        holdout_frac=args.holdout_frac,
        min_history=args.min_history,
        window_rfqs=args.window,
        window_stride=args.stride,
        seed=args.seed,
    )

    # manifest is shared by both backends and by downstream eval
    manifest, _holdout = corpus.build_manifest(df, pcfg)
    manifest, dropped = corpus.apply_min_history(manifest, pcfg.min_history)
    cell_counts = manifest["cell"].value_counts().to_dict()
    n_holdout = manifest[manifest["is_holdout"]]["client_id"].nunique()
    n_seen = manifest[~manifest["is_holdout"]]["client_id"].nunique()
    print(f"      cells={cell_counts}")
    print(f"      seen={n_seen} holdout={n_holdout} dropped={dropped}")

    print(f"[5/5] writing artifacts (backend={args.backend}) ...")
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    if args.backend == "gpu":
        # packed token-ID corpus: no string assembly, no load-time parse.
        special_ids = {"<bos>": tok.bos_id, "<sep>": tok.sep_id, "<eos>": tok.eos_id}
        tokens, offsets = corpus.build_pretrain_corpus_packed(
            manifest, mat, special_ids, pcfg, n_fields=tok.n_fields
        )
        n_seq = len(offsets) - 1
        perm = rng.permutation(n_seq)
        n_val = int(n_seq * args.val_frac)
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        def _subpack(idx):
            # rebuild contiguous packed arrays for a subset of sequences
            lens = offsets[idx + 1] - offsets[idx]
            new_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            buf = np.empty(int(new_off[-1]), dtype=np.int32)
            for j, k in enumerate(idx):
                buf[new_off[j]:new_off[j + 1]] = tokens[offsets[k]:offsets[k + 1]]
            return buf, new_off

        tr_tok, tr_off = _subpack(train_idx)
        va_tok, va_off = _subpack(val_idx)
        np.save(out / "cellA_train_tokens.npy", tr_tok)
        np.save(out / "cellA_train_offsets.npy", tr_off)
        np.save(out / "cellA_val_tokens.npy", va_tok)
        np.save(out / "cellA_val_offsets.npy", va_off)
        print(f"      packed: {len(train_idx):,} train / {len(val_idx):,} val seqs "
              f"({len(tr_tok):,}+{len(va_tok):,} tokens)")
    else:
        lines_list = corpus.build_pretrain_corpus(manifest, mat, tok.id_to_token, pcfg)
        lines = np.array(lines_list, dtype=object)
        perm = rng.permutation(len(lines))
        n_val = int(len(lines) * args.val_frac)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        with open(out / "cellA_train.txt", "w") as f:
            f.write("\n".join(lines[train_idx].tolist()))
        with open(out / "cellA_val.txt", "w") as f:
            f.write("\n".join(lines[val_idx].tolist()))
        print(f"      text: {len(train_idx):,} train / {len(val_idx):,} val lines")

    with open(out / "rfq_vocab.json", "w") as f:
        json.dump(tok.get_state(), f)

    man = manifest.reset_index(drop=True)
    keep_cols = ["client_id", "week_bucket", "is_test", "is_holdout", "cell", "time_rank"]
    man[keep_cols].to_parquet(out / "manifest.parquet", index=True)

    print(f"      vocab + manifest saved under {args.out}/")
    print("done.")


if __name__ == "__main__":
    main()
