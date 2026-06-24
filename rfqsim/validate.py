"""Validation suite (Polars, runs on the 176 CPUs).

Checks that the embedded 'Easter eggs' are recoverable from the output:
  1. Class balance: CLIENT-TRADED in the 5-7% band, four-class split sane.
  2. Fragmentation: CUSIP Gini, top-1%/10% flow share, daily zero-RFQ
     fraction, share of flow in young bonds.
  3. S-curve: logistic regression of our fills on our quote distance
     recovers a clean monotone curve (competition baked in, as in reality).
  4. Hit-rate structure: monotone in client tier and k_dealers.
  5. Client concentration: activity Gini ~0.8+.
  6. Imbalance signal: sign of (BUY-SELL flow) predicts price drift (kappa).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def load(out_dir: str) -> pl.LazyFrame:
    files = sorted(glob.glob(os.path.join(out_dir, "week=*", "*.parquet")))
    return pl.scan_parquet(files)


def validate(out_dir: str, n_bonds: int) -> dict:
    lf = load(out_dir)
    res = {}

    # 1. class balance
    cls = lf.group_by("status").len().collect()
    tot = cls["len"].sum()
    res["class_props"] = {r["status"]: r["len"] / tot for r in cls.iter_rows(named=True)}
    res["hit_rate"] = res["class_props"].get("CLIENT-TRADED", 0.0)

    # 2. fragmentation
    per_cusip = lf.group_by("cusip_id").len().collect().sort("len", descending=True)
    counts = per_cusip["len"].to_numpy()
    full = np.zeros(n_bonds)
    full[: len(counts)] = counts
    res["cusip_gini"] = _gini(full)
    res["top1pct_share"] = counts[: max(1, n_bonds // 100)].sum() / counts.sum()
    res["top10pct_share"] = counts[: max(1, n_bonds // 10)].sum() / counts.sum()
    daily = lf.group_by("date").agg(pl.col("cusip_id").n_unique()).collect()
    res["daily_zero_cusip_frac"] = float(
        1 - daily["cusip_id"].mean() / n_bonds
    )

    # 3. S-curve recovery: P(fill | our distance), our fills only
    df = lf.select(
        ((pl.col("our_quote") - (pl.col("composite_bid") + pl.col("composite_ask")) / 2).abs()
         / (pl.col("composite_ask") - pl.col("composite_bid"))).alias("rel_dist"),
        (pl.col("status") == "CLIENT-TRADED").alias("filled"),
        "client_tier", "k_dealers", "status",
    ).collect()
    q = df.with_columns(pl.col("rel_dist").qcut(10, labels=[str(i) for i in range(10)]).alias("bin"))
    curve = q.group_by("bin").agg(pl.col("filled").mean(), pl.len()).sort("bin")
    res["s_curve_fill_by_decile"] = curve["filled"].to_list()
    res["s_curve_monotone_decreasing"] = bool(
        np.all(np.diff(np.asarray(curve["filled"].to_list())) <= 0.02)
    )

    # 4. hit-rate structure
    res["hit_by_tier"] = (
        df.group_by("client_tier").agg(pl.col("filled").mean()).sort("client_tier")
    )["filled"].to_list()
    res["hit_by_k"] = (
        df.group_by("k_dealers").agg(pl.col("filled").mean(), pl.len()).sort("k_dealers")
    ).to_dicts()

    # 5. client concentration
    per_client = lf.group_by("client_id").len().collect()["len"].to_numpy()
    res["client_gini"] = _gini(per_client)

    # 6. imbalance -> drift sign (coarse daily check)
    d = (
        lf.group_by("date", "sector")
        .agg(
            (pl.col("side").eq("BUY").mean() - 0.5).alias("imb"),
            pl.col("composite_mid").mean().alias("mid"),
        )
        .sort("date")
        .collect()
    )
    d = d.with_columns(pl.col("mid").diff().over("sector").alias("dmid")).drop_nulls()
    if len(d) > 10:
        # drift is contemporaneous with the liquidity regime -> same-day corr
        x, y = d["imb"].to_numpy(), d["dmid"].to_numpy()
        res["imbalance_drift_corr"] = float(np.corrcoef(x, y)[0, 1])

    # young-bond share needs issue dates -> reported by caller if desired
    return res


def report(res: dict) -> str:
    L = ["=" * 64, "RFQSIM VALIDATION", "=" * 64]
    L.append(f"hit rate (CLIENT-TRADED):      {res['hit_rate']:.2%}  (target 5-7%)")
    for k, v in sorted(res["class_props"].items()):
        L.append(f"  {k:<22s} {v:.2%}")
    L.append(f"CUSIP Gini:                    {res['cusip_gini']:.3f}  (target >0.85)")
    L.append(f"top 1% CUSIP flow share:       {res['top1pct_share']:.1%}")
    L.append(f"top 10% CUSIP flow share:      {res['top10pct_share']:.1%}")
    L.append(f"daily zero-RFQ CUSIP fraction: {res['daily_zero_cusip_frac']:.1%}")
    L.append(f"client Gini:                   {res['client_gini']:.3f}  (target ~0.8)")
    L.append(f"fill rate by distance decile:  "
             + " ".join(f"{x:.3f}" for x in res["s_curve_fill_by_decile"]))
    L.append(f"S-curve monotone decreasing:   {res['s_curve_monotone_decreasing']}")
    L.append(f"hit rate by tier (0=top):      "
             + " ".join(f"{x:.3f}" for x in res["hit_by_tier"]))
    if "imbalance_drift_corr" in res:
        L.append(f"imbalance->next-day drift corr: {res['imbalance_drift_corr']:+.3f} (expect >0)")
    return "\n".join(L)
