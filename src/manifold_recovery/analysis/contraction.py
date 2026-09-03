"""Manifold contraction metric V(c, g), class count N_H, and H1-H3 tests.

V            = certified fraction of K decoded candidates (the paper metric).
V_proposal   = certified fraction of K PROPOSAL samples (B4 acceptance),
               the normalisation denominator separating physical
               recoverability from model quality (feasibility review R4).
N_H          = number of side-signature classes among certified decodes.

H3 joins V against the PUBLISHED ICRA success grid (Fig. 5), embedded below,
so the lost 320-trial bags are not needed for the initial analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import Config
from ..scenario import ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID
from ..traj.rtp import RTP
from ..score.obstacles import DockField
from ..certify.surrogate import certify_batch
from ..certify.cluster import side_signature
from ..data.proposal import ProposalSampler

# ICRA Fig. 5 mission success (%) : {(sea_state, degradation_pct): success}
ICRA_SUCCESS = {
    (2, 50): 100, (2, 75): 100, (2, 95): 75, (2, 100): 15,
    (3, 50): 100, (3, 75): 100, (3, 95): 100, (3, 100): 50,
    (4, 50): 100, (4, 75): 100, (4, 95): 100, (4, 100): 60,
    (5, 50): 100, (5, 75): 100, (5, 95): 90, (5, 100): 55,
}


def compute_maps(decode_fn, cfg: Config, sampler, rng,
                 severities=(0.5, 0.75, 0.95, 1.0),
                 sea_states=(2, 3, 4, 5), directions=("nominal",),
                 K: int = 100, n_env: int = 3,
                 x0=SPIKE_START_POSE, zone_id: int = SPIKE_ZONE_ID) -> pd.DataFrame:
    """decode_fn(c_scalar_h1, K) -> omega (K, 2Bw); pass None to skip V and
    compute only V_proposal (useful before training)."""
    rtp = RTP(cfg.trajectory)
    field = DockField()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng)
    zone = ZONES[zone_id]
    T_h, dt = rtp.horizon(x0, zone)
    rows = []
    for deg in severities:
        h1 = 1.0 - deg
        for ss in sea_states:
            for dirn in directions:
                Vs, Vps, Ns = [], [], []
                for _ in range(n_env):
                    w, sig = sampler.sample(ss, dirn, T_h, dt, rng)
                    w = w[:cfg.trajectory.N]
                    wb = lambda om: np.broadcast_to(w, (len(om),) + w.shape)
                    if decode_fn is not None:
                        om = decode_fn(h1, K)
                        cert = certify_batch(om, x0, zone, h1, 1.0, wb(om),
                                             sig, rtp, field, cfg)
                        Vs.append(cert.mask.mean())
                        ok = np.flatnonzero(cert.mask)
                        if len(ok):
                            kin = rtp.kinematics(om[ok], x0, zone)
                            sigs = side_signature(kin.pos, kin.vel)
                            Ns.append(len({tuple(s) for s in sigs.reshape(-1, sigs.shape[-1])}))
                        else:
                            Ns.append(0)
                    omp = prop.sample(K, rng)
                    certp = certify_batch(omp, x0, zone, h1, 1.0, wb(omp),
                                          sig, rtp, field, cfg)
                    Vps.append(certp.mask.mean())
                rows.append({
                    "degradation": deg, "h1": h1, "sea_state": ss,
                    "direction": dirn,
                    "V": float(np.mean(Vs)) if Vs else np.nan,
                    "V_proposal": float(np.mean(Vps)),
                    "N_H": float(np.mean(Ns)) if Ns else np.nan,
                })
    return pd.DataFrame(rows)


def h1_test(df: pd.DataFrame):
    """H1: V decreases with severity (per sea state, then pooled)."""
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
            xs.append(r.V)
            ys.append(ICRA_SUCCESS[key])
    if len(xs) < 4:
        return {"spearman": np.nan, "p": np.nan, "pass": False, "n": len(xs)}
    rho, p = spearmanr(xs, ys)
    return {"spearman": float(rho), "p": float(p), "pass": rho > 0.5,
            "n": len(xs)}
