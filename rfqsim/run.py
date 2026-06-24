#!/usr/bin/env python3
"""Run the RFQ simulator.

Full year at production scale (auto-detects GPUs):
    python run.py --out ./rfq_out

Smoke test (CPU, small):
    python run.py --smoke --out ./rfq_smoke

Seed ensemble for model benchmarking (where 4x H200 earns its keep):
    python run.py --seeds 16 --out ./rfq_ensemble
"""
from __future__ import annotations

import argparse

from rfqsim.config import SimConfig
from rfqsim.orchestrate import run_ensemble, run_simulation
from rfqsim.validate import report, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./rfq_out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=0, help="run an ensemble of N seeds")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--rfqs-per-day", type=float, default=None)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--cpu-workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--ig-desk",
        action="store_true",
        help="sell-side IG corp desk preset: 50k CUSIPs, ~1667 issuers, 10k clients",
    )
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfg = SimConfig()
    cfg.run.seed = args.seed
    if args.smoke:
        cfg = cfg.scaled(bonds=3000, clients=800, days=10, rfqs_per_day=20_000)
        cfg.run.n_cpu_workers = 2
    elif args.ig_desk:
        # Widened instrument universe for a sell-side IG corp bond desk.
        # bonds//30 => ~1667 issuers preserves the ~30-bonds/issuer Zipf head,
        # keeping issuer-level flow concentration in-spec while CUSIP breadth
        # and silent-tail fraction become desk-realistic. Client axis is
        # orthogonal (client latents never reference n_bonds).
        cfg = cfg.scaled(bonds=50_000, clients=10_000)
    if args.days:
        cfg.flow.trading_days = args.days
    if args.rfqs_per_day:
        cfg.flow.rfqs_per_day_target = args.rfqs_per_day
    if args.no_gpu:
        cfg.run.use_gpu = False
    if args.cpu_workers:
        cfg.run.n_cpu_workers = args.cpu_workers

    if args.seeds:
        run_ensemble(cfg, args.seeds, args.out)
        return

    run_simulation(cfg, out_dir=args.out)
    if not args.no_validate:
        res = validate(args.out, cfg.universe.n_bonds)
        print(report(res))


if __name__ == "__main__":
    main()
