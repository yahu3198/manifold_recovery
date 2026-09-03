"""Typed configuration for the whole pipeline (spec Section 11).

Every artifact (dataset, checkpoint, maps) stores ``Config.hash()`` so results
are always traceable to the configuration that produced them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrajectoryCfg:
    N: int = 80
    Bw: int = 20                 # number of basis functions (spec's B; renamed vs thruster matrix)
    alpha_basis: float = 50.0
    ramp_eps: float = 0.1
    v_nom: float = 1.0
    T_min: float = 40.0
    T_max: float = 150.0
    # "straight": T_h = |p_g - p_0| / v_nom (original spike behaviour).
    # "detour":   T_h = max(straight, longest DETOUR_POLYLINES arc) / v_nom, so
    #             around-dock classes are traversable at v_nom and severe faults
    #             are not forced to the straight-line speed (ICRA 95% cases ran
    #             at ~0.4 m/s). Changes the config hash: regenerate from 01.
    horizon_mode: str = "straight"


@dataclass(frozen=True)
class WrenchCfg:
    u_max: float = 2353.0
    l: float = 2.05427
    alpha_bar: float = 1.0
    rho_u_frac: float = 0.05
    rho_a: float = 0.05
    rho_growth_T: float = 60.0
    eps_cert: float = 25.0       # N^2 on max_k (r_x^2 + r_psi^2): thruster axes only
    sway_drift_frac: float = 0.5 # sway criterion: sum_k |r_y,k| / |Y_v| * dt (the uncorrected
                                 # lateral drift the sway residual would cause) must stay
                                 # below sway_drift_frac * exact.e_max. Twin thrusters cannot
                                 # produce sway at ANY health, so the sway residual is a
                                 # sideslip demand, not an actuator infeasibility; it is
                                 # bounded here and arbitrated by the tier-2 rollout.
    pgd_iters: int = 300
    cert_iters: int = 1500
    R0: float = 0.001
    gamma_R: float = 0.01
    kappa: float = 1.5
    eps_conf: float = 0.01


@dataclass(frozen=True)
class ScoreCfg:
    lambda_o: float = 1.0
    lambda_s: float = 0.1
    lambda_f: float = 10.0
    lambda_e: float = 0.01
    eps_obs: float = 3.0


@dataclass(frozen=True)
class DataCfg:
    n_samples: int = 8000
    h1_range: tuple = (0.05, 1.0)
    h1_holdout: tuple = (0.20, 0.30)
    h2_fixed: float = 1.0
    sea_state: int = 3
    direction: str = "nominal"
    chunk: int = 512
    seed: int = 0
    prop_mid_std_m: float = 3.0
    # Proposal mixture weights over (straight, north detour, south detour) bases.
    # (1, 0, 0) reproduces the original single-mode STOMP proposal.
    prop_mix: tuple = (0.6, 0.2, 0.2)
    shape_per_mode: bool = True   # score shaping within (decile x mode); False = rev 1 global


@dataclass(frozen=True)
class ModelCfg:
    latent_dim: int = 1
    hidden: tuple = (256, 256)
    gamma: float = 10.0
    Cz_max: float = 5.0
    a_shaping: float = 10.0
    lr: float = 1e-3
    batch: int = 256
    epochs: int = 700
    recon_sigma: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class OnlineCfg:
    K: int = 100
    z_lo: float = -1.64
    z_hi: float = 1.64
    k_present: int = 3
    beta_max_deg: float = 25.0


@dataclass(frozen=True)
class PlannerCfg:
    N_ocp: int = 40
    ipopt_max_iter: int = 400
    ipopt_retries: int = 2       # warm re-solves from the last iterate on non-convergence
    slack_tol: float = 0.05      # m; total dock slack above this = constraint-violating
    dock_margin: float = 1.5
    w_slack: float = 1e4


@dataclass(frozen=True)
class ExactCfg:
    dt_sim: float = 0.1
    kp: float = 0.2
    kd: float = 0.9
    kpsi: float = 0.4
    kr: float = 1.2
    e_max: float = 5.0
    arrive_tol: float = 1.5


@dataclass(frozen=True)
class Config:
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    wrench: WrenchCfg = field(default_factory=WrenchCfg)
    score: ScoreCfg = field(default_factory=ScoreCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    online: OnlineCfg = field(default_factory=OnlineCfg)
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    exact: ExactCfg = field(default_factory=ExactCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _coerce(v):
    """Lists -> tuples; numeric-looking strings -> float (PyYAML 1.1 parses
    '1.0e4' as a string because the exponent sign is mandatory)."""
    if isinstance(v, list):
        return tuple(_coerce(x) for x in v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _tupled(d: dict) -> dict:
    return {k: _coerce(v) for k, v in d.items()}


def load(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        trajectory=TrajectoryCfg(**_tupled(raw.get("trajectory", {}))),
        wrench=WrenchCfg(**_tupled(raw.get("wrench", {}))),
        score=ScoreCfg(**_tupled(raw.get("score", {}))),
        data=DataCfg(**_tupled(raw.get("data", {}))),
        model=ModelCfg(**_tupled(raw.get("model", {}))),
        online=OnlineCfg(**_tupled(raw.get("online", {}))),
        planner=PlannerCfg(**_tupled(raw.get("planner", {}))),
        exact=ExactCfg(**_tupled(raw.get("exact", {}))),
    )
