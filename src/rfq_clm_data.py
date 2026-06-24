# SPDX-License-Identifier: Apache-2.0
"""
Bond-RFQ CLM dataset for decoder-only pretraining.

Adapted from NVIDIA's src/clm_data.py. The essential difference: our corpus
lines are ALREADY tokenized into field-token strings (e.g. "CT_2 TIER_0 ...")
by the RFQ tokenizer, so "encoding" a line is a pure string->global-id lookup
against the saved RFQ vocab. There is no re-tokenization and no GPU dependency
at data-loading time.

Corpus line format (produced by src/corpus.py):
    <bos> CT_2 TIER_0 SEC_2 MAT_2 AGE_2 SIDE_1 SZ_19 K_3 REG_0 <sep> ... <eos>

(input_ids, labels) are identical with <pad> positions set to -100, exactly as
in the reference recipe; the HF causal LM forward shifts internally.

NeMo AutoModel wires this in via YAML:
    dataset:
      _target_: src/rfq_clm_data.py:build_rfq_clm_dataset
      data_path: data/rfq_corpus/cellA_train.txt
      vocab_path: data/rfq_corpus/rfq_vocab.json
      seq_length: 4096
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class RFQCLMDataset(Dataset):
    """Causal-LM dataset over pre-tokenized RFQ corpus lines."""

    def __init__(
        self,
        sequences: List[List[int]],
        seq_length: int = 4096,
        pad_token_id: int = 0,
    ):
        self.sequences = sequences
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        if len(tokens) > self.seq_length:
            tokens = tokens[: self.seq_length]

        input_ids = np.full(self.seq_length, self.pad_token_id, dtype=np.int64)
        input_ids[: len(tokens)] = tokens

        labels = np.full(self.seq_length, -100, dtype=np.int64)
        labels[: len(tokens)] = tokens

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }


def _encode_line(line: str, token_to_id: Dict[str, int], unk_id: int) -> List[int]:
    """Map a space-separated token-string line to a list of global ids."""
    return [token_to_id.get(tok, unk_id) for tok in line.split()]


def load_corpus_and_encode(
    data_path: Union[str, Path],
    vocab_path: Union[str, Path],
    seq_length: int = 4096,
) -> RFQCLMDataset:
    """Read corpus lines + saved RFQ vocab, return an RFQCLMDataset."""
    vocab_path = Path(vocab_path)
    with open(vocab_path) as f:
        state = json.load(f)
    token_to_id: Dict[str, int] = {k: int(v) for k, v in state["token_to_id"].items()}
    pad_id = token_to_id["<pad>"]
    unk_id = token_to_id["<unk>"]

    data_path = Path(data_path)
    print(f"Loading RFQ corpus from {data_path} (vocab={len(token_to_id)}) ...")

    sequences: List[List[int]] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sequences.append(_encode_line(line, token_to_id, unk_id))

    print(f"  Loaded {len(sequences):,} sequences (seq_length={seq_length})")
    return RFQCLMDataset(sequences=sequences, seq_length=seq_length, pad_token_id=pad_id)


def build_rfq_clm_dataset(
    data_path: str,
    vocab_path: str,
    seq_length: int = 4096,
    **kwargs,
) -> RFQCLMDataset:
    """NeMo AutoModel entry point (extra YAML keys arrive as kwargs, ignored)."""
    return load_corpus_and_encode(
        data_path=data_path, vocab_path=vocab_path, seq_length=seq_length
    )


# ---------------------------------------------------------------------------
# Packed-corpus dataset (GPU/production path)
# ---------------------------------------------------------------------------


class RFQPackedCLMDataset(Dataset):
    """CLM dataset over a packed (tokens, offsets) corpus — no parsing at load.

    tokens  : 1-D int32 array (all sequences concatenated)
    offsets : 1-D int64 array, sequence k = tokens[offsets[k]:offsets[k+1]]
    """

    def __init__(self, tokens, offsets, seq_length: int = 4096, pad_token_id: int = 0):
        self.tokens = tokens
        self.offsets = offsets
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, idx: int):
        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        toks = self.tokens[a:b]
        if len(toks) > self.seq_length:
            toks = toks[: self.seq_length]
        input_ids = np.full(self.seq_length, self.pad_token_id, dtype=np.int64)
        input_ids[: len(toks)] = toks
        labels = np.full(self.seq_length, -100, dtype=np.int64)
        labels[: len(toks)] = toks
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }


def build_rfq_packed_clm_dataset(
    tokens_path: str,
    offsets_path: str,
    pad_token_id: int = 0,
    seq_length: int = 4096,
    **kwargs,
) -> RFQPackedCLMDataset:
    """NeMo AutoModel entry point for the packed corpus."""
    tokens = np.load(tokens_path)
    offsets = np.load(offsets_path)
    print(f"Loaded packed corpus: {len(offsets)-1:,} sequences, "
          f"{len(tokens):,} tokens from {tokens_path}")
    return RFQPackedCLMDataset(tokens, offsets, seq_length=seq_length, pad_token_id=pad_token_id)
