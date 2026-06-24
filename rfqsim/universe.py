"""Static + slowly-varying universe: issuers, bonds, clients.

Design notes (matching the agreed framework):
- CUSIP activity weight is COMPOSED, not drawn: issuer Zipf x size^a x
  age-decay x curve-point boost x lognormal noise. With a live primary
  calendar, this produces the fragmentation signature (Gini ~0.85+, most
  CUSIPs silent on most days) and makes the skew itself nonstationary.
- Clients are an attribution layer: P(client | bond bucket, side, week).
  Bucket = (sector, maturity bucket, age bucket). Weights = activity a_c(t)
  x type-bucket affinity x mandate mask x side tilt. a_c follows a weekly
  log-OU; mandates are sampled once at onboarding.
- Everything time-varying is piecewise-constant per week: workers rebuild
  small CDF tables per week chunk, so sampling is searchsorted on the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig

MAT_BUCKETS = np.array([2.0, 5.0, 7.0, 10.0, 30.0])
N_AGE_BUCKETS = 3  # <0.5y, 0.5-2y, >2y
AGE_EDGES_DAYS = np.array([0.0, 126.0, 504.0, 1e9])


@dataclass
class Universe:
    cfg: SimConfig
    # issuers
    issuer_sector: np.ndarray = field(default=None)
    issuer_weight: np.ndarray = field(default=None)
    # bonds
    bond_issuer: np.ndarray = field(default=None)
    bond_sector: np.ndarray = field(default=None)
    bond_maturity: np.ndarray = field(default=None)
    bond_mat_bucket: np.ndarray = field(default=None)
    bond_amount: np.ndarray = field(default=None)
    bond_issue_day: np.ndarray = field(default=None)  # trading-day units, can be <0
    bond_noise: np.ndarray = field(default=None)
    bond_spread: np.ndarray = field(default=None)     # composite full spread delta0
    bond_kappa: np.ndarray = field(default=None)
    bond_base_price: np.ndarray = field(default=None)
    bond_sigma: np.ndarray = field(default=None)      # cusip idio vol
    issuer_sigma: np.ndarray = field(default=None)
    sector_bonds: list = field(default=None)          # bond ids per sector
    # clients
    client_type: np.ndarray = field(default=None)     # int index into cfg types
    client_tier: np.ndarray = field(default=None)     # 0,1,2
    client_base_activity: np.ndarray = field(default=None)
    client_intent: np.ndarray = field(default=None)
    client_side_tilt: np.ndarray = field(default=None)   # P(buy-side RFQ tilt)
    client_mandate_sector: np.ndarray = field(default=None)  # bool (C, S)
    client_mandate_mat: np.ndarray = field(default=None)     # bool (C, 5)
    type_age_affinity: np.ndarray = field(default=None)      # (T, 3)
    type_mat_affinity: np.ndarray = field(default=None)      # (T, 5)
    client_activity_weekly: np.ndarray = field(default=None) # (W, C) log-OU path
    n_weeks: int = 0


def build_universe(cfg: SimConfig, rng: np.random.Generator) -> Universe:
    u = Universe(cfg=cfg)
    uc, fc = cfg.universe, cfg.flow

    # ---------------- issuers ----------------
    nI, nS = uc.n_issuers, uc.n_sectors
    u.issuer_sector = rng.integers(0, nS, size=nI)
    ranks = np.arange(1, nI + 1, dtype=np.float64)
    w = ranks ** (-uc.issuer_zipf_a)
    rng.shuffle(w)
    u.issuer_weight = w / w.sum()
    u.issuer_sigma = (
        cfg.price.sigma_issuer_daily * np.exp(rng.normal(0, 0.25, nI))
    ).astype(np.float64)

    # ---------------- bonds ----------------
    nB = uc.n_bonds
    # bonds per issuer roughly proportional to issuer size
    p_iss = (u.issuer_weight ** 0.7)
    p_iss /= p_iss.sum()
    u.bond_issuer = rng.choice(nI, size=nB, p=p_iss)
    u.bond_sector = u.issuer_sector[u.bond_issuer]

    # maturities clustered at benchmark tenors with scatter (orphans exist)
    mb = rng.choice(len(MAT_BUCKETS), size=nB, p=[0.18, 0.27, 0.13, 0.27, 0.15])
    scatter = rng.normal(0, 0.18, nB)
    u.bond_maturity = MAT_BUCKETS[mb] * np.exp(scatter)
    on_benchmark = np.abs(scatter) < 0.08
    u.bond_mat_bucket = mb.astype(np.int8)

    # amount outstanding: lognormal with an index-eligible lump at >= $1bn
    amt = np.exp(rng.normal(19.6, 0.85, nB))           # median ~ $300m
    big = rng.random(nB) < 0.22
    amt[big] = np.exp(rng.normal(21.0, 0.35, big.sum()))  # ~ $1.3bn deals
    u.bond_amount = amt

    # issue dates: back-fill + live primary calendar through the sim year
    horizon = float(fc.trading_days)
    n_new = rng.poisson(uc.new_issues_per_week * horizon / 5.0)
    n_new = min(n_new, nB // 6)
    pre = nB - n_new
    pre_days = -rng.uniform(0, uc.pre_history_years * 252.0, pre)
    new_days = np.sort(rng.uniform(0, horizon, n_new))
    u.bond_issue_day = np.concatenate([pre_days, new_days])
    perm = rng.permutation(nB)
    u.bond_issue_day = u.bond_issue_day[perm]

    u.bond_noise = np.exp(rng.normal(0, uc.beta_lognoise_sigma, nB))
    u.bond_noise *= np.where(on_benchmark, uc.curve_point_boost, 1.0)

    # pricing statics
    pc = cfg.price
    u.bond_base_price = rng.normal(pc.base_price_mean, pc.base_price_sigma, nB)
    u.bond_kappa = np.clip(rng.normal(pc.kappa_mean, pc.kappa_sigma, nB), 0.01, None)
    u.bond_sigma = pc.sigma_cusip_daily * np.exp(rng.normal(0, 0.3, nB))
    size_z = (np.log(amt) - 19.6) / 0.85
    u.bond_spread = (
        pc.spread_base
        * np.exp(-0.18 * size_z)                       # bigger deal -> tighter
        * np.exp(rng.normal(0, 0.25, nB))
        * (1.0 + 0.12 * np.maximum(u.bond_maturity - 5.0, 0) / 25.0)
    )
    u.sector_bonds = [np.where(u.bond_sector == s)[0] for s in range(nS)]

    # ---------------- clients ----------------
    nC, nT = uc.n_clients, len(uc.client_types)
    u.client_type = rng.choice(nT, size=nC, p=np.asarray(uc.client_type_probs))
    act = (1.0 + rng.pareto(uc.client_activity_pareto_a, nC))
    u.client_base_activity = act / act.mean()
    # tiers correlate with size: big accounts get the relationship edge
    q = np.argsort(-u.client_base_activity)
    tier = np.full(nC, 2, dtype=np.int8)
    tier[q[: int(0.10 * nC)]] = 0
    tier[q[int(0.10 * nC): int(0.40 * nC)]] = 1
    u.client_tier = tier
    ip = cfg.outcome.intent_prob_by_type
    u.client_intent = np.array(
        [ip[uc.client_types[t]] for t in u.client_type]
    ) * np.exp(rng.normal(0, 0.06, nC))
    u.client_intent = np.clip(u.client_intent, 0.2, 0.98)
    # side tilt: insurers net buyers, index two-way, HF noisy
    base_tilt = np.array([0.52, 0.50, 0.60, 0.50, 0.46])
    u.client_side_tilt = np.clip(
        base_tilt[u.client_type] + rng.normal(0, 0.07, nC), 0.15, 0.85
    )

    # mandates
    m_sec = rng.random((nC, nS)) < uc.mandate_sector_frac
    m_sec[np.arange(nC), rng.integers(0, nS, nC)] = True  # at least one sector
    u.client_mandate_sector = m_sec
    m_mat = rng.random((nC, 5)) < 0.75
    m_mat[np.arange(nC), rng.integers(0, 5, nC)] = True
    # insurers: force long-end in mandate
    ins = u.client_type == 2
    m_mat[ins, 3] = True
    m_mat[ins, 4] = True
    u.client_mandate_mat = m_mat

    # type affinities (log-scale): HF loves new issues, insurer long & seasoned
    u.type_age_affinity = np.array([
        [0.3, 0.1, 0.0],    # asset_mgr
        [0.9, 0.1, -0.4],   # hedge_fund
        [-0.2, 0.0, 0.3],   # insurer
        [0.5, 0.2, -0.1],   # index
        [0.2, 0.0, 0.0],    # bank
    ])
    u.type_mat_affinity = np.array([
        [0.0, 0.2, 0.1, 0.2, 0.1],
        [0.2, 0.3, 0.1, 0.2, -0.2],
        [-0.5, -0.1, 0.0, 0.4, 0.7],
        [0.0, 0.2, 0.0, 0.3, 0.2],
        [0.4, 0.2, 0.0, -0.1, -0.3],
    ])

    # weekly log-OU activity path (precomputed => reproducible across workers)
    n_weeks = int(np.ceil(fc.trading_days / 5.0))
    u.n_weeks = n_weeks
    hl = uc.activity_ou_halflife_weeks
    phi = 0.5 ** (1.0 / hl)
    sig = uc.activity_ou_sigma
    x = rng.normal(0, sig / np.sqrt(1 - phi ** 2), nC)
    path = np.empty((n_weeks, nC), dtype=np.float32)
    for wk in range(n_weeks):
        x = phi * x + rng.normal(0, sig, nC)
        path[wk] = x
    u.client_activity_weekly = path
    return u


# ---------------------------------------------------------------------------
# Weekly tables consumed by the engine (piecewise-constant within a week)
# ---------------------------------------------------------------------------

def bond_weights_for_week(u: Universe, week_day0: float) -> np.ndarray:
    """Unnormalised CUSIP flow weights at the start of a week (length n_bonds).

    beta_i ~ issuer_w x amount^a x age_decay x noise; zero pre-issuance.
    """
    uc = u.cfg.universe
    age = week_day0 - u.bond_issue_day
    live = age >= 0
    hl = uc.age_decay_halflife_days
    decay = uc.age_floor + (1 - uc.age_floor) * np.exp(-np.log(2) * np.maximum(age, 0) / hl)
    w = (
        u.issuer_weight[u.bond_issuer]
        * (u.bond_amount / 3e8) ** uc.size_flow_exp
        * decay
        * u.bond_noise
    )
    w[~live] = 0.0
    return w


def bond_age_bucket(u: Universe, week_day0: float) -> np.ndarray:
    age = np.maximum(week_day0 - u.bond_issue_day, 0.0)
    return (np.searchsorted(AGE_EDGES_DAYS, age, side="right") - 1).clip(0, 2).astype(np.int8)


def client_bucket_cdfs(u: Universe, week_idx: int):
    """CDF over clients for each (sector, mat_bucket, age_bucket, side).

    Returns (cdf, order): cdf shape (S, 5, 3, 2, C) float32, order = client ids
    (identity; kept for clarity). Sampling: searchsorted(cdf[s,m,a,side], u).
    Memory: 8*5*3*2*5000*4B = 4.8 MB. Rebuilt per week in each worker.
    """
    cfg = u.cfg
    nC = cfg.universe.n_clients
    nS = cfg.universe.n_sectors
    a_c = u.client_base_activity * np.exp(
        u.client_activity_weekly[min(week_idx, u.n_weeks - 1)]
    )
    t = u.client_type
    aff_mat = u.type_mat_affinity[t]                     # (C, 5)
    aff_age = u.type_age_affinity[t]                     # (C, 3)
    tilt = u.client_side_tilt                            # buy prob
    side_fac = np.stack([1 - tilt, tilt], axis=1)        # (C, 2): [bid(sell), ask(buy)]

    # weight tensor (C, 5, 3): activity x exp(affinities) x mandate
    w = (
        a_c[:, None, None]
        * np.exp(aff_mat)[:, :, None]
        * np.exp(aff_age)[:, None, :]
        * u.client_mandate_mat[:, :, None]
    )
    cdf = np.empty((nS, 5, 3, 2, nC), dtype=np.float32)
    for s in range(nS):
        ws = w * u.client_mandate_sector[:, s][:, None, None]
        for side in range(2):
            full = ws * side_fac[:, side][:, None, None]      # (C,5,3)
            c = np.cumsum(full, axis=0)                       # cumulate over clients
            tot = c[-1]                                       # (5,3)
            tot = np.where(tot <= 0, 1.0, tot)
            cdf[s, :, :, side] = np.transpose(c / tot, (1, 2, 0))
    return cdf
