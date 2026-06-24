"""Configuration for the RFQ simulator.

Defaults target: 5k clients, 30k IG CUSIPs, ~100k RFQs/day, 1 trading year,
~5-7% desk hit rate, heavily fragmented CUSIP activity.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MMPPConfig:
    # Two intensity levels per side (low/high), exchangeable 4-state joint chain
    # (LL, LH, HL, HH). Rates per trading day, magnitudes echo Bergault-Gueant
    # Table 1, rescaled so sector totals hit the target daily RFQ count.
    lam_low_frac: float = 0.62    # low state as fraction of sector mean intensity
    lam_high_frac: float = 2.10   # high state as fraction of sector mean intensity
    # Symmetric generator rates (per day): see mmpp.build_generator
    r_sym_to_asym: float = 4.0    # LL->LH (= LL->HL), HH->LH (= HH->HL)
    r_sym_to_sym: float = 1.2     # LL->HH and HH->LL
    r_asym_to_sym_low: float = 16.0   # LH->LL
    r_asym_to_sym_high: float = 9.0   # LH->HH
    r_asym_flip: float = 3.0      # LH->HL (jump to opposite imbalance)


@dataclass
class UniverseConfig:
    n_sectors: int = 8
    n_issuers: int = 1000
    n_bonds: int = 30_000
    n_clients: int = 5_000
    issuer_zipf_a: float = 1.25          # issuer size power law
    size_flow_exp: float = 1.30          # flow ~ amount_outstanding^a
    age_decay_halflife_days: float = 120.0
    age_floor: float = 0.12              # seasoned-bond residual activity
    beta_lognoise_sigma: float = 0.55    # idiosyncratic CUSIP noise
    benchmark_maturities: tuple = (2.0, 5.0, 7.0, 10.0, 30.0)
    curve_point_boost: float = 1.6       # extra weight near benchmark tenors
    new_issues_per_week: float = 30.0    # primary calendar intensity
    pre_history_years: float = 8.0       # back-fill of existing issue dates
    client_activity_pareto_a: float = 1.10
    client_types: tuple = ("asset_mgr", "hedge_fund", "insurer", "index", "bank")
    client_type_probs: tuple = (0.42, 0.18, 0.16, 0.12, 0.12)
    mandate_sector_frac: float = 0.65    # fraction of sectors in a client mandate
    activity_ou_halflife_weeks: float = 10.0
    activity_ou_sigma: float = 0.18      # weekly log-activity shock scale


@dataclass
class FlowConfig:
    rfqs_per_day_target: float = 100_000.0
    trading_days: int = 252
    day_start_hour: float = 7.0
    day_hours: float = 10.0
    intraday_u_shape: float = 0.35       # 0 = flat, >0 = open/close humps


@dataclass
class PriceConfig:
    # Issuer-level spread-curve factor; CUSIPs priced off their issuer.
    sigma_issuer_daily: float = 0.18     # $ / sqrt(day) on price points
    sigma_cusip_daily: float = 0.06      # idiosyncratic CUSIP wiggle
    kappa_mean: float = 0.45             # drift per unit imbalance (paper Table 2 range)
    kappa_sigma: float = 0.35
    base_price_mean: float = 99.5
    base_price_sigma: float = 4.0
    spread_base: float = 0.45            # composite half-spread scale ($)
    spread_age_mult: float = 1.8         # seasoned bonds: wider
    spread_size_mult: float = 0.6        # benchmark deals: tighter


@dataclass
class OutcomeConfig:
    # Auction decomposition: trade-at-all (logistic on best quote) x win (best of k).
    logit_alpha: float = -0.70           # paper-calibrated S-curve
    logit_beta: float = 3.10
    dealers_in_comp: dict = field(default_factory=lambda: {
        "asset_mgr": (4, 8), "hedge_fund": (3, 6), "insurer": (3, 6),
        "index": (5, 8), "bank": (2, 5),
    })
    large_size_k_decrement: int = 2      # big tickets go to fewer dealers
    large_size_threshold: float = 5e6
    our_quote_noise: float = 0.22        # rel. dispersion of our half-spread
    competitor_noise: float = 0.38
    tier_edge: tuple = (0.05, 0.0, -0.05)  # tier-1/2/3 price edge in half-spreads
    regime_skew: float = 0.35            # defensive skew in imbalanced states
    intent_prob_by_type: dict = field(default_factory=lambda: {
        "asset_mgr": 0.80, "hedge_fund": 0.72, "insurer": 0.85,
        "index": 0.90, "bank": 0.55,
    })
    cancel_share_no_trade: float = 0.30  # CANCELLED vs EXPIRED split baseline
    cancel_vol_sensitivity: float = 0.25
    size_lognorm_mu: float = 13.6        # ~ $800k median
    size_lognorm_sigma: float = 1.05
    odd_lot_floor: float = 100_000.0
    disclose_cover_on_traded_away: bool = True


@dataclass
class RunConfig:
    out_dir: str = "./rfq_out"
    seed: int = 7
    n_gpu_workers: int = 0               # 0 => auto (device count, else CPU procs)
    n_cpu_workers: int = 8               # parquet writers / fallback sim procs
    week_chunk_days: int = 5
    use_gpu: bool = True
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 3
    float_dtype: str = "float32"


@dataclass
class SimConfig:
    mmpp: MMPPConfig = field(default_factory=MMPPConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    price: PriceConfig = field(default_factory=PriceConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def scaled(self, bonds=None, clients=None, days=None, rfqs_per_day=None):
        """Convenience for smoke tests."""
        import copy

        c = copy.deepcopy(self)
        if bonds:
            c.universe.n_bonds = bonds
            c.universe.n_issuers = max(20, bonds // 30)
        if clients:
            c.universe.n_clients = clients
        if days:
            c.flow.trading_days = days
        if rfqs_per_day:
            c.flow.rfqs_per_day_target = rfqs_per_day
        return c
