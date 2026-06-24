#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Thin launcher: delegates to NeMo AutoModel's training recipe.

Mirrors the reference TFM launcher. On the GPU box this resolves the YAML
(_target_ fields, dataset builder, FSDP2) and runs the loop. Kept as a separate
entry point so the RFQ config and the RFQ CLM data builder are picked up.

Usage:
  python scripts/train_decoder_model.py \
      -c configs/pretrain_bond_rfq.yaml \
      --dataset.data_path data/rfq_corpus/cellA_train.txt \
      --dataset.vocab_path data/rfq_corpus/rfq_vocab.json \
      --validation_dataset.data_path data/rfq_corpus/cellA_val.txt \
      --validation_dataset.vocab_path data/rfq_corpus/rfq_vocab.json
"""
from nemo_automodel.recipes.llm.pretrain import main  # type: ignore

if __name__ == "__main__":
    main()
