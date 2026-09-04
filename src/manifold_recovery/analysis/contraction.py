"""Manifold contraction metric V(c), lift V / V_proposal, class count N_H, H1-H3 (rev 3).

V            = certified fraction of K decoded candidates (paper metric).
V_proposal   = certified fraction of K NAIVE proposal samples (B4 acceptance),
               the normaliser separating physical recoverability from what
               the manifold added (feasibility review R4).
lift         = V / V_proposal.
N_H          = number of (zone, winding) classes among certified decodes.

Maps are computed at the CANONICAL start (scenario.SPIKE_START_POSE) for
comparability with the ICRA grid; H3 joins against the published Fig. 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import Config
from ..scenario import SPIKE_START_POSE
from ..traj.rtp import RTP
from ..score.obstacles import DockField
from ..certify.surrogate import certify_batch
from ..certify.cluster import side_signature
from ..features.condition import build_c
from ..data.proposal import ProposalSampler

ICRA_SUCCESS = {
    (2, 50): 100, (2, 75): 100, (2, 95): 75, (2, 100): 15,
    (3, 50): 100, (3, 75): 100, (3, 95): 100, (3, 100): 50,
    (4, 50): 100, (4, 75): 100, (4, 95): 100, (4, 100): 60,
    (5, 50): 100, (5, 75): 100, (5, 95): 90, (5, 100): 55,
}


def n_classes(pos: np.ndarray) -> int:
    if len(pos) == 0:
        return 0
    sgs = side_signature(pos)
    return len({tuple(s) for s in sgs.reshape(-1, sgs.shape[-1]).tolist()})


def compute_maps(decoder, cfg: Config, sampler, rng,
                 severities=(0.5, 0.75, 0.95, 1.0),
                 sea_states=(2, 3, 4, 5), directions=("nominal",),
                 K: int = 100, n_env: int = 3, x0=SPIKE_START_POSE) -> pd.DataFrame:
    """decoder: model.decode.Decoder (or None for proposal-only maps)."""
    rtp = RTP(cfg.trajectory)
    field = DockField()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, mix=cfg.data.prop_mix)
    T_h, dt = rtp.horizon()
    x0 = np.asarray(x0, float)
    rows = []
    for deg in severities:
        h1 = 1.0 - deg
        c = build_c(h1, x0, spike=True)
        for ss in sea_states:
            for dirn in directions:
                Vs, Vps, Ns = [], [], []
                for _ in range(n_env):
                    w, sig = sampler.sample(ss, dirn, T_h, dt, rng)
                    w = w[:cfg.trajectory.N]
                    wb = lambda om: np.broadcast_to(w, (len(om),) + w.shape)
                    if decoder is not None:
                        om = decoder.decode(c, K, rng)
                        cert = certify_batch(om, x0, h1, 1.0, wb(om), sig, rtp, field, cfg)
                        Vs.append(cert.mask.mean())
                        ok = np.flatnonzero(cert.mask)
                        Ns.append(n_classes(rtp.kinematics(om[ok], x0).pos) if len(ok) else 0)
                    omp = prop.sample(x0, rng, K)
                    certp = certify_batch(omp, x0, h1, 1.0, wb(omp), sig, rtp, field, cfg)
                    Vps.append(certp.mask.mean())
                V = float(np.mean(Vs)) if Vs else np.nan
                Vp = float(np.mean(Vps))
                rows.append({"degradation": deg, "h1": h1, "sea_state": ss,
                             "direction": dirn, "V": V, "V_proposal": Vp,
                             "lift": V / Vp if Vp > 0 else np.nan,
                             "N_H": float(np.mean(Ns)) if Ns else np.nan})
    return pd.DataFrame(rows)


def h1_test(df: pd.DataFrame):
    """H1: V decreases with severity (pooled over sea states)."""
    rho, p = spearmanr(df["degradation"], df["V"])
    return {"spearman": float(rho), "p": float(p), "pass": rho < -0.5}


def h2_test(df: pd.DataFrame, deg: float = 1.0):
    """H2: at near-total failure, moderate seas beat calm seas (SS4 > SS2)."""
    sel = df[df["degradation"] == deg]
    v2 = float(sel[sel.sea_state == 2]["V"].mean())
    v4 = float(sel[sel.sea_state == 4]["V"].mean())
    return {"V_ss2": v2, "V_ss4": v4, "pass": v4 > v2}


def h3_test(df: pd.DataFrame):
    """H3: V correlates with the published ICRA closed-loop success grid."""
    xs, ys = [], []
    for _, r in df.iterrows():
        key = (int(r.sea_state), int(round(r.degradation * 100)))
        if key in ICRA_SUCCESS and np.isfinite(r.V):
            xs.append(r.V); ys.append(ICRA_SUCCESS[key])
    if len(xs) < 4:
        return {"spearman": np.nan, "p": np.nan, "pass": False, "n": len(xs)}
    rho, p = spearmanr(xs, ys)
    return {"spearman": float(rho), "p": float(p), "pass": rho > 0.5, "n": len(xs)}


def lift_test(df: pd.DataFrame, min_lift: float = 1.5, headroom: float = 0.6):
    """Pre-registered rev-3 criterion: lift >= min_lift in at least half of the
    cells where the proposal is starved (V_proposal < headroom)."""
    sel = df[(df["V_proposal"] < headroom) & np.isfinite(df["lift"])]
    if len(sel) == 0:
        return {"n_cells": 0, "frac_pass": np.nan, "pass": False}
    frac = float((sel["lift"] >= min_lift).mean())
    return {"n_cells": int(len(sel)), "frac_pass": frac, "median_lift": float(sel["lift"].median()),
            "pass": frac >= 0.5}
