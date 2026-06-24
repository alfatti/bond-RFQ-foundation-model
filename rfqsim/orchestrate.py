"""Orchestration: shard the year across GPU workers, write partitioned Parquet.

Parallel decomposition
----------------------
- Time is the shard axis: each worker owns a set of week-chunks. Chunks are
  independent because (a) MMPP chains + the weekly issuer-factor grid are
  precomputed on the master and shipped to workers, and (b) intra-week prices
  are Brownian *bridges* between grid points -> paths are continuous across
  chunk boundaries regardless of which worker generated them.
- One process per GPU (RFQSIM_DEVICE pins the CuPy device). With no GPU,
  the same workers run on NumPy across CPU processes.
- Parquet writes (zstd) happen inside workers; with 4 workers on a fat NVMe
  box, generation is compute-bound on GPU and I/O-bound on CPU.

Scale notes for 4x H200 / 176 cores / 740 GB:
- 25M rows/yr is a few GB: a single H200 generates it in well under a minute;
  the box's real leverage is seed ensembles (--seeds N runs N independent
  years round-robin across GPUs) for model benchmarking.
"""
from __future__ import annotations

import os
import time
import multiprocessing as mp

import numpy as np

from .backend import device_count, get_xp, pinned_rng, set_device
from .config import SimConfig
from .engine import STATUS_NAMES, generate_chunk
from .mmpp import interval_table, simulate_sector_chain
from .universe import Universe, bond_weights_for_week, build_universe, client_bucket_cdfs


def _build_shared_state(cfg: SimConfig, seed: int):
    rng = np.random.default_rng(seed)
    u = build_universe(cfg, rng)
    T = float(cfg.flow.trading_days)
    per_sector = cfg.flow.rfqs_per_day_target / cfg.universe.n_sectors / 2.0
    chains, tables = [], []
    for s in range(cfg.universe.n_sectors):
        ch = simulate_sector_chain(cfg.mmpp, per_sector, T, rng)
        chains.append(ch)
        tables.append(interval_table(ch, T))
    # weekly issuer-factor grid (continuous prices across chunks)
    n_weeks = u.n_weeks
    nI = cfg.universe.n_issuers
    grid = np.zeros((n_weeks + 1, nI), dtype=np.float64)
    for w in range(n_weeks):
        d = min(5.0, T - w * 5.0)
        grid[w + 1] = grid[w] + rng.standard_normal(nI) * u.issuer_sigma * np.sqrt(max(d, 1e-9))
    return u, chains, tables, grid


def _worker(args):
    (cfg, seed, weeks, u, tables, chain_pack, grid, device, out_dir) = args
    os.environ["RFQSIM_DEVICE"] = str(device)
    if cfg.run.use_gpu:
        try:
            set_device(device)
        except Exception:
            pass
    xp = get_xp(cfg.run.use_gpu)
    import pyarrow as pa
    import pyarrow.parquet as pq

    T = float(cfg.flow.trading_days)
    nS = cfg.universe.n_sectors
    dates = np.busday_offset(np.datetime64("2025-01-06"), np.arange(cfg.flow.trading_days))
    rows_written = 0

    for wk in weeks:
        t0, t1 = wk * 5.0, min(wk * 5.0 + 5.0, T)
        if t1 <= t0:
            continue
        rng = pinned_rng(xp, seed * 1_000_003 + wk * 97 + device)
        bw = bond_weights_for_week(u, t0)
        ccdf = client_bucket_cdfs(u, wk)
        parts = []
        for s in range(nS):
            tab = tables[s]
            sel = (tab[:, 1] > t0) & (tab[:, 0] < t1)
            iv = tab[sel]
            if len(iv) == 0:
                continue
            ct, ci, cr = chain_pack[s]
            out = generate_chunk(
                xp, rng, u, s, wk, t0, t1, iv,
                bw[u.sector_bonds[s]], u.sector_bonds[s],
                ccdf[s], grid[wk], grid[wk + 1],
                ct, cr, ci,
                rfq_id_offset=(wk * 100 + s) * 10_000_000,
            )
            if out is not None:
                parts.append(out)
        if not parts:
            continue
        chunk = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

        # wall-clock timestamps with intraday U-shape
        td = np.clip(chunk.pop("t_days"), 0, T - 1e-9)
        day = np.floor(td).astype(np.int64)
        x = td - day
        ushape = cfg.flow.intraday_u_shape
        x = np.clip(x - (ushape / (2 * np.pi)) * np.sin(2 * np.pi * x), 0, 1)
        secs = (cfg.flow.day_start_hour + cfg.flow.day_hours * x) * 3600.0
        ts = dates[np.clip(day, 0, len(dates) - 1)].astype("datetime64[s]") \
            + secs.astype("timedelta64[s]")
        order = np.argsort(ts, kind="stable")

        tbl = pa.table({
            "rfq_id": chunk["rfq_id"][order],
            "timestamp": ts[order],
            "date": dates[np.clip(day, 0, len(dates) - 1)][order],
            "sector": chunk["sector"][order],
            "mat_bucket": chunk["mat_bucket"][order],
            "age_bucket": chunk["age_bucket"][order],
            "issuer_id": chunk["issuer_id"][order],
            "cusip_id": chunk["cusip_id"][order],
            "client_id": chunk["client_id"][order],
            "client_type": chunk["client_type"][order],
            "client_tier": chunk["client_tier"][order],
            "side": np.where(chunk["side"][order] == 1, "BUY", "SELL"),
            "size": chunk["size"][order],
            "k_dealers": chunk["k_dealers"][order],
            "composite_bid": chunk["composite_bid"][order],
            "composite_ask": chunk["composite_ask"][order],
            "composite_mid": chunk["composite_mid"][order],
            "our_quote": chunk["our_quote"][order],
            "cover_price": chunk["cover_price"][order],
            "mmpp_state": chunk["mmpp_state"][order],
            "status": STATUS_NAMES[chunk["status"][order]],
        })
        path = os.path.join(out_dir, f"week={wk:03d}")
        os.makedirs(path, exist_ok=True)
        pq.write_table(
            tbl, os.path.join(path, "part-0.parquet"),
            compression=cfg.run.parquet_compression,
            compression_level=cfg.run.parquet_compression_level,
        )
        rows_written += tbl.num_rows
    return rows_written


def run_simulation(cfg: SimConfig, seed: int | None = None, out_dir: str | None = None) -> int:
    seed = cfg.run.seed if seed is None else seed
    out_dir = out_dir or cfg.run.out_dir
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    u, chains, tables, grid = _build_shared_state(cfg, seed)
    chain_pack = [
        (c.times, c.imb_integral, c.lam_a[c.states] - c.lam_b[c.states]) for c in chains
    ]
    n_weeks = u.n_weeks

    n_gpu = device_count() if cfg.run.use_gpu else 0
    n_workers = (cfg.run.n_gpu_workers or n_gpu) if n_gpu else max(1, cfg.run.n_cpu_workers)
    n_workers = min(n_workers, n_weeks)
    shards = [list(range(w, n_weeks, n_workers)) for w in range(n_workers)]
    args = [
        (cfg, seed, shards[w], u, tables, chain_pack, grid,
         (w % n_gpu) if n_gpu else 0, out_dir)
        for w in range(n_workers)
    ]

    if n_workers == 1:
        totals = [_worker(args[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            totals = pool.map(_worker, args)

    total = int(sum(totals))
    dt = time.time() - t_start
    print(f"[rfqsim] {total:,} RFQs -> {out_dir} | {n_workers} worker(s) "
          f"({'GPU' if n_gpu else 'CPU'}) | {dt:.1f}s | {total/max(dt,1e-9):,.0f} rows/s")
    return total


def run_ensemble(cfg: SimConfig, n_seeds: int, base_out: str):
    """Seed ensemble: where 4x H200 actually pays off. Each seed is a full
    independent year; seeds are distributed round-robin across GPUs."""
    for s in range(n_seeds):
        run_simulation(cfg, seed=cfg.run.seed + s, out_dir=os.path.join(base_out, f"seed={s:03d}"))
