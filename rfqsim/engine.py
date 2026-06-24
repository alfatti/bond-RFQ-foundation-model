"""Hot-path generation engine. Everything here runs on `xp` (CuPy or NumPy).

Per (week, sector) chunk:
  1. Poisson RFQ counts per MMPP interval x side -> scatter uniform timestamps
     (no thinning; O(total RFQs)).
  2. CUSIP sampling: searchsorted on the weekly within-sector weight CDF.
  3. Client sampling: searchsorted per (mat_bucket, age_bucket, side) combo
     on the weekly client CDFs (30 small batched groups, GPU-friendly).
  4. Composite mid: weekly issuer-factor grid + Brownian bridge inside the
     week (exact, chunk-independent => price paths are continuous across
     workers) + kappa * sector imbalance integral (exact, piecewise linear)
     + CUSIP idio dispersion.
  5. Outcomes: auction decomposition -> TRADED / TRADED-AWAY / CANCELLED /
     EXPIRED with closed-form best-of-(k-1) cover sampling.
"""
from __future__ import annotations

import numpy as np

from .backend import to_np
from .universe import Universe, bond_age_bucket

STATUS_TRADED = 0
STATUS_TRADED_AWAY = 1
STATUS_CANCELLED = 2
STATUS_EXPIRED = 3
STATUS_NAMES = np.array(
    ["CLIENT-TRADED", "CLIENT-TRADED-AWAY", "CLIENT-CANCELLED", "EXPIRED"]
)


def _ndtri(xp, p):
    """Inverse standard normal CDF on either backend."""
    if xp is np:
        from scipy.special import erfinv
        return np.sqrt(2.0) * erfinv(2.0 * p - 1.0)
    from cupyx.scipy.special import erfinv  # type: ignore
    return xp.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def _segmented_cumsum(xp, values, seg_first_pos):
    """Inclusive cumsum within segments. seg_first_pos[i] = index where
    element i's segment starts (elements sorted by segment)."""
    cs = xp.cumsum(values)
    offset = xp.where(
        seg_first_pos > 0, cs[xp.maximum(seg_first_pos - 1, 0)], 0.0
    )
    return cs - offset


def generate_chunk(
    xp,
    rng,
    u: Universe,
    sector: int,
    week_idx: int,
    week_t0: float,
    week_t1: float,
    intervals: np.ndarray,          # rows of mmpp.interval_table clipped to week
    bond_w_sector: np.ndarray,      # weekly weights for this sector's bonds
    sector_bond_ids: np.ndarray,
    client_cdf_sector: np.ndarray,  # (5, 3, 2, C) float32 for this sector
    issuer_grid_a: np.ndarray,      # issuer factor at week start (nI,)
    issuer_grid_b: np.ndarray,      # issuer factor at week end   (nI,)
    chain_times: np.ndarray,        # sector chain transition times
    chain_imb_rate: np.ndarray,     # lam_a - lam_b per interval
    chain_imb_int: np.ndarray,      # cumulative integral at chain_times
    rfq_id_offset: int,
):
    cfg = u.cfg
    oc, pc = cfg.outcome, cfg.price
    f32 = xp.float32

    # ---------------- 1. counts + timestamps ----------------
    t0 = xp.asarray(np.maximum(intervals[:, 0], week_t0))
    t1 = xp.asarray(np.minimum(intervals[:, 1], week_t1))
    dur = xp.clip(t1 - t0, 0.0, None)
    lam = xp.asarray(intervals[:, 3:5])                     # (n_int, 2) bid/ask
    counts = rng.poisson(lam * dur[:, None]).astype(xp.int64)   # (n_int, 2)

    n = int(counts.sum())
    if n == 0:
        return None
    flat = counts.reshape(-1)
    iv = xp.repeat(xp.arange(flat.shape[0]), flat)          # interval*2+side idx
    side = (iv % 2).astype(xp.int8)                         # 0=bid(client sells)
    ivi = iv // 2
    ts = t0[ivi] + rng.random(n) * dur[ivi]

    # ---------------- 2. CUSIP sampling ----------------
    wsec = xp.asarray(bond_w_sector, dtype=xp.float64)
    cdf_b = xp.cumsum(wsec)
    cdf_b /= cdf_b[-1]
    loc = xp.searchsorted(cdf_b, rng.random(n), side="right")
    loc = xp.clip(loc, 0, len(sector_bond_ids) - 1)
    sb = xp.asarray(sector_bond_ids)
    bond = sb[loc]

    # ---------------- 3. client sampling ----------------
    mb = xp.asarray(u.bond_mat_bucket)[bond].astype(xp.int64)
    ab = xp.asarray(bond_age_bucket(u, week_t0))[bond].astype(xp.int64)
    combo = (mb * 3 + ab) * 2 + side.astype(xp.int64)       # 0..29
    cdf_c = xp.asarray(client_cdf_sector)                   # (5,3,2,C)
    cdf_flat = cdf_c.reshape(30, -1)
    client = xp.empty(n, dtype=xp.int64)
    uc = rng.random(n)
    for g in range(30):
        m = combo == g
        cnt = int(m.sum())
        if cnt:
            client[m] = xp.searchsorted(cdf_flat[g], uc[m], side="right")
    client = xp.clip(client, 0, cfg.universe.n_clients - 1)

    # ---------------- 4. composite mid ----------------
    issuer = xp.asarray(u.bond_issuer)[bond]
    order = xp.lexsort(xp.stack([ts, issuer]))              # sort by issuer, ts
    inv = xp.empty_like(order)
    inv[order] = xp.arange(n)
    iss_s, ts_s = issuer[order], ts[order]
    first = xp.searchsorted(iss_s, iss_s, side="left")
    prev_t = xp.where(first == xp.arange(n), week_t0, xp.concatenate([ts_s[:1], ts_s[:-1]]))
    dt = xp.clip(ts_s - prev_t, 1e-9, None)
    sig_i = xp.asarray(u.issuer_sigma)[iss_s]
    incr = rng.standard_normal(n) * sig_i * xp.sqrt(dt)
    W = _segmented_cumsum(xp, incr, first)
    # total Brownian displacement over the full week per segment (for bridge)
    last = xp.searchsorted(iss_s, iss_s, side="right") - 1
    tail_dt = xp.clip(week_t1 - ts_s[last], 1e-9, None)
    seg_tail = rng.standard_normal(n) * sig_i * xp.sqrt(tail_dt)  # draw per elem,
    W_tot = W[last] + seg_tail[first]                       # use segment-first draw
    Xa = xp.asarray(issuer_grid_a)[iss_s]
    Xb = xp.asarray(issuer_grid_b)[iss_s]
    frac = (ts_s - week_t0) / (week_t1 - week_t0)
    bridge = Xa + W - frac * (W_tot - (Xb - Xa))
    issuer_factor = bridge[inv]

    # exact MMPP imbalance integral at ts
    ct = xp.asarray(chain_times)
    k = xp.clip(xp.searchsorted(ct, ts, side="right") - 1, 0, len(chain_times) - 1)
    imb_int = xp.asarray(chain_imb_int)[k] + xp.asarray(chain_imb_rate)[k] * (ts - ct[k])
    imb_rate_now = xp.asarray(chain_imb_rate)[k]            # for skew/cancel logic

    kappa = xp.asarray(u.bond_kappa)[bond]
    idio = rng.standard_normal(n) * xp.asarray(u.bond_sigma)[bond]
    mid = (
        xp.asarray(u.bond_base_price)[bond]
        + issuer_factor
        + kappa * imb_int / xp.asarray(u.cfg.flow.rfqs_per_day_target / 8.0)
        + idio
    )

    # ---------------- 5. outcomes ----------------
    d0 = xp.asarray(u.bond_spread)[bond]                    # full composite spread
    half = 0.5 * d0
    tier = xp.asarray(u.client_tier)[client].astype(xp.int64)
    tier_edge = xp.asarray(np.asarray(oc.tier_edge))[tier]
    imb_sign = xp.sign(imb_rate_now)
    # dealer skews quotes toward the imbalance: defensive on the hot side
    skew = oc.regime_skew * half * imb_sign * xp.where(side == 1, 1.0, -1.0)
    d_us = half * (1.0 + oc.our_quote_noise * rng.standard_normal(n)) + skew \
        - tier_edge * half
    d_us = xp.clip(d_us, 0.02 * d0, None)

    size = xp.maximum(
        xp.exp(rng.standard_normal(n) * oc.size_lognorm_sigma + oc.size_lognorm_mu),
        oc.odd_lot_floor,
    )
    ctype = xp.asarray(u.client_type)[client]
    k_lo = xp.asarray(np.array([oc.dealers_in_comp[t][0] for t in cfg.universe.client_types]))
    k_hi = xp.asarray(np.array([oc.dealers_in_comp[t][1] for t in cfg.universe.client_types]))
    kd = (k_lo[ctype] + (rng.random(n) * (k_hi[ctype] - k_lo[ctype] + 1)).astype(xp.int64))
    kd = xp.where(size > oc.large_size_threshold, kd - oc.large_size_k_decrement, kd)
    kd = xp.clip(kd, 1, None)

    # best cover among (k-1) iid quotes: min quantile p ~ Beta(1, m)
    m_comp = xp.clip(kd - 1, 0, None).astype(f32)
    u_min = rng.random(n)
    p_min = 1.0 - (1.0 - u_min) ** (1.0 / xp.maximum(m_comp, 1.0))
    z_min = _ndtri(xp, xp.clip(p_min, 1e-7, 1 - 1e-7))
    d_cov = half * (1.0 + oc.competitor_noise * z_min)
    d_cov = xp.where(m_comp < 1, xp.inf, xp.clip(d_cov, 0.02 * d0, None))

    d_best = xp.minimum(d_us, d_cov)
    intent = rng.random(n) < xp.asarray(u.client_intent)[client]
    p_trade = 1.0 / (1.0 + xp.exp(oc.logit_alpha + oc.logit_beta * d_best / d0))
    traded = intent & (rng.random(n) < p_trade)
    we_win = d_us <= d_cov

    status = xp.full(n, STATUS_EXPIRED, dtype=xp.int8)
    status[traded & we_win] = STATUS_TRADED
    status[traded & ~we_win] = STATUS_TRADED_AWAY
    no_trade = ~traded
    p_cancel = (
        oc.cancel_share_no_trade
        + 0.30 * (~intent)
        + oc.cancel_vol_sensitivity * (imb_sign != 0)
    )
    cancelled = no_trade & (rng.random(n) < p_cancel)
    status[cancelled] = STATUS_CANCELLED

    sgn = xp.where(side == 1, 1.0, -1.0)
    our_quote = mid + sgn * d_us
    cover = xp.where(
        (status == STATUS_TRADED_AWAY) & oc.disclose_cover_on_traded_away,
        mid + sgn * d_cov,
        xp.nan,
    )

    out = {
        "rfq_id": np.arange(rfq_id_offset, rfq_id_offset + n, dtype=np.int64),
        "t_days": to_np(ts).astype(np.float64),
        "sector": np.full(n, sector, dtype=np.int8),
        "mat_bucket": to_np(mb).astype(np.int8),
        "age_bucket": to_np(ab).astype(np.int8),
        "mmpp_state": to_np(xp.asarray(intervals[:, 2].astype(np.int8))[ivi]),
        "issuer_id": to_np(issuer).astype(np.int32),
        "cusip_id": to_np(bond).astype(np.int32),
        "client_id": to_np(client).astype(np.int32),
        "client_type": to_np(ctype).astype(np.int8),
        "client_tier": to_np(tier).astype(np.int8),
        "side": to_np(side),                                # 0=client sells, 1=buys
        "size": to_np(size).astype(np.float64),
        "k_dealers": to_np(kd).astype(np.int8),
        "composite_mid": to_np(mid).astype(np.float64),
        "composite_bid": to_np(mid - half).astype(np.float64),
        "composite_ask": to_np(mid + half).astype(np.float64),
        "our_quote": to_np(our_quote).astype(np.float64),
        "cover_price": to_np(cover).astype(np.float64),
        "status": to_np(status),
    }
    return out
