#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Lift evaluation harness for the Bond-RFQ foundation model.

Measures the headline scientific result: how much the FM client-state embedding
adds OVER an auction-features-only baseline for win prediction.

    baseline   = XGBoost on live-RFQ auction mechanics only
    augmented  = XGBoost on [auction features || FM embedding]
    LIFT       = augmented_metric - baseline_metric

Trained on Cell A (seen clients, train weeks); evaluated on:
    Cell B  (seen clients, future weeks)  -> HEADLINE (deployment-realistic)
    Cell D  (novel clients, future weeks) -> honesty check (generalization)

If Cell B lift is strong but Cell D lift collapses, the embedding is memorising
client identities rather than learning transferable client state.

The embedding for each scored RFQ is point-in-time (strictly-prior history),
produced by src/extract_embeddings.py and joined here by (client_id, time_rank).

Auction baseline features (live RFQ, decision-time):
    quote_distance, rel_spread, k_dealers, log_size, side,
    sector, mat_bucket, age_bucket, mmpp_state
Deliberately EXCLUDES client_type/client_tier/client_id so the embedding must
earn the lift by encoding client latents beyond static attributes.

Label: won = (status == 'CLIENT-TRADED').

Usage:
  python eval/run_lift_eval.py \
      --rfq-dir data/igdesk_dev \
      --corpus-dir data/rfq_corpus \
      --embeddings data/rfq_corpus/embeddings.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Auction feature engineering (live-RFQ, decision-time)
# ---------------------------------------------------------------------------

AUCTION_NUM = ["quote_distance", "rel_spread", "log_size"]
AUCTION_CAT = ["k_dealers", "side_code", "sector", "mat_bucket", "age_bucket", "mmpp_state"]


def build_auction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the auction baseline feature frame from raw RFQ columns."""
    mid = df["composite_mid"].to_numpy()
    q = df["our_quote"].to_numpy()
    bid = df["composite_bid"].to_numpy()
    ask = df["composite_ask"].to_numpy()
    side_code = (df["side"].to_numpy() == "BUY").astype(np.int64)

    # signed quote distance: positive = more aggressive toward winning.
    # For a client BUY (desk sells), lower ask is more aggressive; for a client
    # SELL (desk buys), higher bid is more aggressive. Encode as quote minus mid
    # with sign so that "better for client" is consistent.
    raw_dist = (q - mid) / np.where(mid != 0, mid, np.nan)
    sign = np.where(side_code == 1, -1.0, 1.0)  # buy: cheaper=better -> flip
    quote_distance = raw_dist * sign

    feats = pd.DataFrame(index=df.index)
    feats["quote_distance"] = quote_distance
    feats["rel_spread"] = (ask - bid) / np.where(mid != 0, mid, np.nan)
    feats["log_size"] = np.log1p(df["size"].to_numpy())
    feats["k_dealers"] = df["k_dealers"].to_numpy()
    feats["side_code"] = side_code
    feats["sector"] = df["sector"].to_numpy()
    feats["mat_bucket"] = df["mat_bucket"].to_numpy()
    feats["age_bucket"] = df["age_bucket"].to_numpy()
    feats["mmpp_state"] = df["mmpp_state"].to_numpy()
    return feats


# ---------------------------------------------------------------------------
# Model training / eval
# ---------------------------------------------------------------------------


def _fit_xgb(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Train an XGBoost classifier (handles the ~5% positive imbalance)."""
    from xgboost import XGBClassifier

    pos = float(y.mean())
    spw = (1.0 - pos) / max(pos, 1e-6)  # scale_pos_weight for imbalance
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
        tree_method="hist",
    )
    clf.fit(X, y)
    return clf


def _metrics(clf, X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    p = clf.predict_proba(X)[:, 1]
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def load_rfq(rfq_dir: str) -> pd.DataFrame:
    import polars as pl

    files = sorted(Path(rfq_dir).glob("week=*/*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files]).to_pandas()
    return df.sort_values(["timestamp", "rfq_id"], kind="stable").reset_index(drop=True)


def assemble(rfq_dir: str, corpus_dir: str, embeddings_path: str):
    df = load_rfq(rfq_dir)
    df["time_rank"] = np.arange(len(df))  # matches corpus stable order
    manifest = pd.read_parquet(Path(corpus_dir) / "manifest.parquet")
    emb = pd.read_parquet(embeddings_path)  # client_id, time_rank, history_len, e0..

    # join manifest cell labels onto RFQ rows by (client_id, time_rank)
    key = ["client_id", "time_rank"]
    base = df.merge(manifest[key + ["cell", "is_test", "is_holdout"]], on=key, how="inner")
    base = base.merge(emb, on=key, how="left")

    feats = build_auction_features(base)
    emb_cols = [c for c in emb.columns if c.startswith("e")]
    y = (base["status"].to_numpy() == "CLIENT-TRADED").astype(np.int64)
    return base, feats, emb_cols, y


def run(args):
    base, feats, emb_cols, y = assemble(args.rfq_dir, args.corpus_dir, args.embeddings)
    base["_y"] = y

    cellA = base["cell"] == "A"
    cellB = base["cell"] == "B"
    cellD = base["cell"] == "D"

    # rows must have an embedding; drop history_len==0 if requested
    has_emb = base[emb_cols].notna().all(axis=1).to_numpy()
    if args.min_history > 0:
        has_emb &= (base["history_len"].fillna(0).to_numpy() >= args.min_history)

    Xauc = feats.to_numpy(dtype=np.float32)
    Xemb = base[emb_cols].to_numpy(dtype=np.float32)
    Xcat = np.concatenate([Xauc, Xemb], axis=1)

    def subset(mask):
        m = mask.to_numpy() & has_emb
        return m

    mA = subset(cellA)
    print(f"train Cell A rows (with embedding): {mA.sum():,} | base rate {y[mA].mean():.3%}")

    # train both models on Cell A
    print("training auction baseline ...")
    clf_auc = _fit_xgb(Xauc[mA], y[mA], seed=args.seed)
    print("training augmented [auction || embedding] ...")
    clf_cat = _fit_xgb(Xcat[mA], y[mA], seed=args.seed)

    rows = []
    for name, mask in [("B (headline: seen/future)", cellB), ("D (honesty: novel/future)", cellD)]:
        m = subset(mask)
        if m.sum() == 0:
            continue
        mb = _metrics(clf_auc, Xauc[m], y[m])
        mc = _metrics(clf_cat, Xcat[m], y[m])
        rows.append((name, mb, mc))

    print()
    print("=" * 74)
    print("LIFT RESULTS  (augmented - auction baseline)")
    print("=" * 74)
    print(f"{'cell':<28}{'n':>9}{'base%':>7}{'AUC_base':>10}{'AUC_aug':>9}{'ΔAUC':>8}{'ΔAP':>8}")
    for name, mb, mc in rows:
        dauc = mc["auc"] - mb["auc"]
        dap = mc["ap"] - mb["ap"]
        print(f"{name:<28}{mb['n']:>9,}{mb['base_rate']*100:>6.1f}{mb['auc']:>10.4f}"
              f"{mc['auc']:>9.4f}{dauc:>+8.4f}{dap:>+8.4f}")
    print()
    print("Interpretation: ΔAUC on Cell B is the value the client-state embedding")
    print("adds at deployment. If Cell D ΔAUC << Cell B ΔAUC, the embedding is")
    print("memorising client identities, not learning transferable state.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfq-dir", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--min-history", type=int, default=5,
                    help="drop scored RFQs with fewer than this many prior RFQs")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
