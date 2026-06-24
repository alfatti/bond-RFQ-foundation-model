#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
GPU parity test — RUN THIS ON THE BREV/H200 BOX before any full run.

Confirms the GPU fast paths match the numpy/CPU correctness oracles on real
hardware. Two checks:

  1. TOKENIZER PARITY (must be BIT-IDENTICAL)
     numpy RFQTokenizer.tokenize_rfqs  ==  cuDF RFQTokenizerGPU.tokenize_rfqs
     Token ids are integers; any difference is a bug. Tolerance = 0.

  2. EXTRACTOR PARITY (must be within bf16 tolerance)
     numpy-forward oracle embeddings  ~=  GPU-gather bf16 embeddings
     bf16 introduces ~1e-2 relative error; we assert max abs diff < 5e-2 and
     cosine similarity > 0.999. A larger gap indicates a gather/indexing bug,
     not just precision.

Exit code 0 = parity holds. Non-zero = investigate before training.

Usage (on the box, with RAPIDS + CUDA torch):
  python scripts/parity_test.py \
      --rfq-dir data/igdesk --corpus-dir data/rfq_corpus \
      --checkpoint models/bond-rfq-fm/checkpoints/<step>   # optional; else random init
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import RFQTokenizer  # noqa: E402
from src.tokenizer.rfq_tokenizer import FIELD_ORDER  # noqa: E402


def load_sample(rfq_dir: str, n: int):
    import cudf
    files = sorted(Path(rfq_dir).glob("week=*/*.parquet"))
    gdf = cudf.read_parquet(files[0])
    gdf = gdf.head(n)
    pdf = gdf.to_pandas()
    return gdf, pdf


def test_tokenizer(gdf, pdf, vocab_state) -> bool:
    from src.tokenizer.rfq_tokenizer_gpu import RFQTokenizerGPU
    import cupy as cp

    ref = RFQTokenizer.from_state(vocab_state)
    gpu = RFQTokenizerGPU(vocab_state)

    cpu_mat = ref.tokenize_rfqs(pdf)              # (n, F) numpy
    gpu_mat = cp.asnumpy(gpu.tokenize_rfqs(gdf))  # (n, F) numpy

    ok = np.array_equal(cpu_mat, gpu_mat)
    print(f"[1] tokenizer parity: bit-identical = {ok}  "
          f"({cpu_mat.shape[0]:,} rows x {cpu_mat.shape[1]} fields)")
    if not ok:
        diff = cpu_mat != gpu_mat
        bad = [FIELD_ORDER[i] for i in np.where(diff.any(axis=0))[0]]
        print(f"    MISMATCH in fields: {bad}")
        r, c = np.argwhere(diff)[0]
        print(f"    first: row {r} field {FIELD_ORDER[c]} cpu={cpu_mat[r,c]} gpu={gpu_mat[r,c]}")
    return ok


def test_extractor(pdf, vocab_state, checkpoint) -> bool:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.extract_embeddings import PointInTimeExtractor, ExtractionConfig, make_gpu_gather

    ref = RFQTokenizer.from_state(vocab_state)
    words = ref.tokenize_rfqs(pdf)
    # one synthetic client over the sample so we exercise long-sequence batching
    n = len(pdf)
    cid = np.zeros(n, dtype=np.int64)
    tr = np.arange(n, dtype=np.int64)

    if checkpoint:
        model = LlamaForCausalLM.from_pretrained(checkpoint)
    else:
        lc = LlamaConfig(vocab_size=ref.vocab_size, hidden_size=128, num_hidden_layers=3,
                         num_attention_heads=8, num_key_value_heads=2, intermediate_size=256,
                         max_position_embeddings=4096, tie_word_embeddings=True,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
        model = LlamaForCausalLM(lc)
    model.eval()
    D = model.config.hidden_size

    @torch.no_grad()
    def oracle_fn(ids2d, pad_id):
        # CPU fp32 oracle
        m = model.to("cpu")
        ids = torch.from_numpy(ids2d); am = (ids != pad_id).long()
        return m(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1].numpy()

    cfg = ExtractionConfig(score_block=128, ctx_carry=64, batch_windows=8, max_seq_len=2048)

    # oracle (CPU fp32)
    ext_oracle = PointInTimeExtractor(oracle_fn, len(FIELD_ORDER), ref.pad_id, ref.bos_id,
                                      ref.sep_id, ref.eos_id, D, cfg=cfg)
    ro = ext_oracle.extract(cid, words, tr)

    # GPU gather (bf16/cuda)
    gather = make_gpu_gather(model, device="cuda", dtype="bfloat16")
    ext_gpu = PointInTimeExtractor(oracle_fn, len(FIELD_ORDER), ref.pad_id, ref.bos_id,
                                   ref.sep_id, ref.eos_id, D, cfg=cfg, gpu_gather=gather)
    rg = ext_gpu.extract(cid, words, tr)

    oo = np.argsort(ro["time_rank"]); og = np.argsort(rg["time_rank"])
    same_rows = np.array_equal(ro["time_rank"][oo], rg["time_rank"][og])
    eo, eg = ro["embedding"][oo], rg["embedding"][og]
    maxdiff = float(np.abs(eo - eg).max())
    cos = float((eo * eg).sum(1).mean() /
                (np.linalg.norm(eo, axis=1) * np.linalg.norm(eg, axis=1) + 1e-9).mean())
    ok = same_rows and maxdiff < 5e-2 and cos > 0.999
    print(f"[2] extractor parity: rows_match={same_rows} max_abs_diff={maxdiff:.2e} "
          f"mean_cos={cos:.5f}  -> {'OK' if ok else 'FAIL'}")
    if not ok and maxdiff >= 5e-2:
        print("    diff exceeds bf16 tolerance -> likely a gather/indexing bug, not precision")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfq-dir", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-sample", type=int, default=20000)
    args = ap.parse_args()

    vocab_state = json.load(open(Path(args.corpus_dir) / "rfq_vocab.json"))
    gdf, pdf = load_sample(args.rfq_dir, args.n_sample)

    ok1 = test_tokenizer(gdf, pdf, vocab_state)
    ok2 = test_extractor(pdf, vocab_state, args.checkpoint)

    print()
    if ok1 and ok2:
        print("PARITY HOLDS — GPU fast paths match the oracles. Safe to run full scale.")
        sys.exit(0)
    print("PARITY FAILED — investigate before committing GPU hours.")
    sys.exit(1)


if __name__ == "__main__":
    main()
