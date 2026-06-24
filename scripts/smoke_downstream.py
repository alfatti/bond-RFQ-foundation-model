#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end CPU smoke test of the full downstream chain:
  1. train a tiny FM on real Cell A corpus (few hundred steps)
  2. extract point-in-time embeddings for a client subset (Cells A/B/D)
  3. run the lift harness (auction baseline vs augmented)

This validates that every component connects and produces sane outputs. A tiny
under-trained model on a 40-day dev slice will NOT show meaningful lift; the
point here is plumbing correctness, not the scientific result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.rfq_clm_data import build_rfq_clm_dataset  # noqa: E402
from src.tokenizer import RFQTokenizer  # noqa: E402
from src.tokenizer.rfq_tokenizer import FIELD_ORDER  # noqa: E402
from src.extract_embeddings import PointInTimeExtractor, ExtractionConfig  # noqa: E402
import src.corpus as corpus  # noqa: E402

CORPUS = "data/rfq_corpus"
RFQDIR = "data/igdesk_dev"
HID = 96
LAYERS = 3
STEPS = 300
SEQLEN = 1024
N_CLIENTS_EVAL = 400   # subset for fast CPU extraction


def train_tiny_fm() -> LlamaForCausalLM:
    print("[1] training tiny FM on real Cell A ...")
    lc = LlamaConfig(
        vocab_size=75, hidden_size=HID, num_hidden_layers=LAYERS,
        num_attention_heads=6, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=4096, tie_word_embeddings=True,
        rope_theta=100000.0, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = LlamaForCausalLM(lc)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"    params: {nparam:,}")
    ds = build_rfq_clm_dataset(f"{CORPUS}/cellA_train.txt", f"{CORPUS}/rfq_vocab.json", seq_length=SEQLEN)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    it = iter(dl); losses = []
    for step in range(STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        out.loss.backward(); opt.step(); opt.zero_grad()
        losses.append(out.loss.item())
        if step % 100 == 0:
            print(f"    step {step:4d}  loss {out.loss.item():.3f}")
    print(f"    final loss {losses[-1]:.3f} (start {losses[0]:.3f})")
    model.eval()
    return model


def extract_for_eval(model: LlamaForCausalLM):
    print("[2] extracting point-in-time embeddings (client subset) ...")
    import polars as pl
    files = sorted(Path(RFQDIR).glob("week=*/*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files]).to_pandas()
    df = df.sort_values(["timestamp", "rfq_id"], kind="stable").reset_index(drop=True)
    df["time_rank"] = np.arange(len(df))

    manifest = pd.read_parquet(f"{CORPUS}/manifest.parquet")
    # pick a subset of clients that appear in cells B or D (so we have eval rows),
    # plus all their rows (for prior context).
    evalcells = manifest[manifest["cell"].isin(["B", "D"])]
    pick = pd.Index(evalcells["client_id"].unique()[:N_CLIENTS_EVAL])
    sub = df[df["client_id"].isin(pick)].copy()
    print(f"    {len(pick)} clients, {len(sub):,} RFQ rows (incl. context)")

    tok = RFQTokenizer.from_state(json.load(open(f"{CORPUS}/rfq_vocab.json")))
    words = tok.tokenize_rfqs(sub)  # (n, F)

    @torch.no_grad()
    def model_fn(ids2d, pad_id):
        ids = torch.from_numpy(ids2d); am = (ids != pad_id).long()
        return model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1].numpy()

    ext = PointInTimeExtractor(
        model_fn, n_fields=len(FIELD_ORDER), pad_id=tok.pad_id, bos_id=tok.bos_id,
        sep_id=tok.sep_id, eos_id=tok.eos_id, embed_dim=HID,
        cfg=ExtractionConfig(score_block=128, ctx_carry=64, batch_windows=8, max_seq_len=SEQLEN),
    )
    res = ext.extract(sub["client_id"].to_numpy(), words, sub["time_rank"].to_numpy())
    print(f"    extracted {len(res['time_rank']):,} embeddings, dim={res['embedding'].shape[1]}")

    emb_df = pd.DataFrame(res["embedding"], columns=[f"e{i}" for i in range(res["embedding"].shape[1])])
    emb_df.insert(0, "history_len", res["history_len"])
    emb_df.insert(0, "time_rank", res["time_rank"])
    emb_df.insert(0, "client_id", res["client_id"])
    out = f"{CORPUS}/embeddings_smoke.parquet"
    emb_df.to_parquet(out, index=False)
    print(f"    wrote {out}")
    return out


def main():
    model = train_tiny_fm()
    emb_path = extract_for_eval(model)
    print("[3] running lift harness ...")
    import subprocess
    subprocess.run([
        sys.executable, "eval/run_lift_eval.py",
        "--rfq-dir", RFQDIR, "--corpus-dir", CORPUS,
        "--embeddings", emb_path, "--min-history", "5",
    ], check=True)


if __name__ == "__main__":
    main()
