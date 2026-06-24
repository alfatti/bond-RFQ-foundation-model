"""Sector-level bidimensional MMPP (the paper's regime layer).

Joint states over (bid, ask) intensity levels, lexicographic:
0 = LL, 1 = LH, 2 = HL, 3 = HH.

Exchangeability (Appendix A.1): the generator is invariant under swapping
bid<->ask, i.e. swapping states 1 and 2. Every asymmetric state can return
to a symmetric one, so the micro-price limit exists and prices don't drift
unboundedly.

This layer is deliberately CPU/NumPy: a year of CTMC transitions per sector
is a few thousand events -- broadcasting it to GPUs would be pure overhead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MMPPConfig

N_STATES = 4
LL, LH, HL, HH = 0, 1, 2, 3


def build_generator(cfg: MMPPConfig) -> np.ndarray:
    """Exchangeable 4x4 generator. Rows sum to zero."""
    Q = np.zeros((4, 4))
    Q[LL, LH] = Q[LL, HL] = cfg.r_sym_to_asym
    Q[LL, HH] = cfg.r_sym_to_sym
    Q[HH, LH] = Q[HH, HL] = cfg.r_sym_to_asym
    Q[HH, LL] = cfg.r_sym_to_sym
    Q[LH, LL] = Q[HL, LL] = cfg.r_asym_to_sym_low
    Q[LH, HH] = Q[HL, HH] = cfg.r_asym_to_sym_high
    Q[LH, HL] = Q[HL, LH] = cfg.r_asym_flip
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def stationary_dist(Q: np.ndarray) -> np.ndarray:
    A = np.vstack([Q.T, np.ones(Q.shape[0])])
    b = np.zeros(Q.shape[0] + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(pi, 0, None) / pi.sum()


@dataclass
class SectorChain:
    """Realised CTMC path for one sector over the whole horizon.

    times:  transition times in trading-day units, times[0] = 0, last = T
    states: state on [times[k], times[k+1])
    lam_b/lam_a: per-state sector-aggregate intensities (RFQs/day)
    imb_integral: cumulative integral of (lam_a - lam_b) at `times`
                  (piecewise linear in between -> exact price drift).
    """

    times: np.ndarray
    states: np.ndarray
    lam_b: np.ndarray
    lam_a: np.ndarray
    imb_integral: np.ndarray

    def imbalance_at(self, t: np.ndarray) -> np.ndarray:
        """I(t) = int_0^t (lam_a - lam_b) ds, vectorised, exact."""
        k = np.searchsorted(self.times, t, side="right") - 1
        k = np.clip(k, 0, len(self.states) - 1)
        imb_rate = self.lam_a[self.states[k]] - self.lam_b[self.states[k]]
        return self.imb_integral[k] + imb_rate * (t - self.times[k])

    def state_at(self, t: np.ndarray) -> np.ndarray:
        k = np.searchsorted(self.times, t, side="right") - 1
        return self.states[np.clip(k, 0, len(self.states) - 1)]


def simulate_sector_chain(
    cfg: MMPPConfig,
    sector_mean_intensity: float,
    horizon_days: float,
    rng: np.random.Generator,
) -> SectorChain:
    Q = build_generator(cfg)
    pi = stationary_dist(Q)

    lam_lo = cfg.lam_low_frac * sector_mean_intensity
    lam_hi = cfg.lam_high_frac * sector_mean_intensity
    lam_b = np.array([lam_lo, lam_lo, lam_hi, lam_hi])  # bid level per state
    lam_a = np.array([lam_lo, lam_hi, lam_lo, lam_hi])  # ask level per state

    times = [0.0]
    states = [int(rng.choice(N_STATES, p=pi))]
    t = 0.0
    while True:
        s = states[-1]
        rate = -Q[s, s]
        t += rng.exponential(1.0 / rate)
        if t >= horizon_days:
            break
        p = Q[s].copy()
        p[s] = 0.0
        p /= p.sum()
        states.append(int(rng.choice(N_STATES, p=p)))
        times.append(t)
    times.append(horizon_days)

    times = np.asarray(times)
    states = np.asarray(states, dtype=np.int8)

    durations = np.diff(times)
    imb_rate = lam_a[states] - lam_b[states]
    imb_integral = np.concatenate([[0.0], np.cumsum(imb_rate * durations)])[:-1]

    return SectorChain(
        times=times[:-1],
        states=states,
        lam_b=lam_b,
        lam_a=lam_a,
        imb_integral=imb_integral,
    )


def interval_table(chain: SectorChain, horizon_days: float) -> np.ndarray:
    """(n_intervals, 5) table: t0, t1, state, lam_b, lam_a.

    The orchestrator draws Poisson RFQ counts per interval x side from this,
    then scatters uniform timestamps -- O(total RFQs), no thinning.
    """
    t0 = chain.times
    t1 = np.append(chain.times[1:], horizon_days)
    s = chain.states.astype(np.int64)
    return np.column_stack([t0, t1, s, chain.lam_b[s], chain.lam_a[s]])
