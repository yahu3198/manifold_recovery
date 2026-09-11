"""Manifold contraction metric V(c), lift V / V_proposal, class count N_H, H1-H3 (rev 4).

Rev 4: the decoder is zone-conditioned, so K decodes are split equally across
the three zones, matching the zone-balanced naive proposal (prop_mix 1:1:1).

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
                 K: int = 100, n_env: int = 3, x0=SPIKE_START_POSE,
                 starts=None, verbose: bool = False) -> pd.DataFrame:
    """decoder: model.decode.Decoder (or None for proposal-only maps).

    Returns ONE ROW PER FORCE DRAW (long format) with columns
    degradation, h1, sea_state, direction, start, draw, V, V_proposal, N_H.
    Use ``aggregate`` for cell means with bootstrap confidence intervals.
    ``starts``: optional (S, 3) array of start poses; default is the
    canonical start only (start index 0).
    """
    rtp = RTP(cfg.trajectory)
    field = DockField()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, mix=cfg.data.prop_mix)
    T_h, dt = rtp.horizon()
    starts = np.asarray(x0, float)[None, :] if starts is None else np.asarray(starts, float)
    rows = []
    for si, xs in enumerate(starts):
        for deg in severities:
            h1 = 1.0 - deg
            for ss in sea_states:
                for dirn in directions:
                    for k in range(n_env):
                        w, sig = sampler.sample(ss, dirn, T_h, dt, rng)
                        w = w[:cfg.trajectory.N]
                        wb = lambda om: np.broadcast_to(w, (len(om),) + w.shape)
                        V, NH = np.nan, np.nan
                        if decoder is not None:
                            om, _ = decoder.decode_zones(h1, xs, K, rng)
                            cert = certify_batch(om, xs, h1, 1.0, wb(om), sig, rtp, field, cfg)
                            V = float(cert.mask.mean())
                            ok = np.flatnonzero(cert.mask)
                            NH = n_classes(rtp.kinematics(om[ok], xs).pos) if len(ok) else 0
                        omp = prop.sample(xs, rng, K)
                        certp = certify_batch(omp, xs, h1, 1.0, wb(omp), sig, rtp, field, cfg)
                        rows.append({"degradation": deg, "h1": h1, "sea_state": ss,
                                     "direction": dirn, "start": si, "draw": k,
                                     "V": V, "V_proposal": float(certp.mask.mean()), "N_H": NH})
                    if verbose:
                        print(f"  start {si} deg {deg:.2f} ss {ss}: done", flush=True)
    return pd.DataFrame(rows)


def _boot_ci(vals, fn=np.mean, n_boot: int = 1000, rng=None, alpha: float = 0.05):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    stats = fn(vals[idx], axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def aggregate(df_draws: pd.DataFrame, by=("degradation", "h1", "sea_state", "direction"),
              n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Cell means over draws (and starts unless 'start' is in ``by``) with 95 %
    bootstrap CIs for V, V_proposal and lift (lift = ratio of cell means,
    bootstrapped over paired draws)."""
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df_draws.groupby(list(by)):
        V, Vp = g["V"].to_numpy(float), g["V_proposal"].to_numpy(float)
        n = len(g)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update({"n_draws": n, "V": float(np.nanmean(V)) if np.isfinite(V).any() else np.nan,
                    "V_proposal": float(np.mean(Vp)),
                    "N_H": float(np.nanmean(g["N_H"])) if np.isfinite(g["N_H"]).any() else np.nan})
        row["V_lo"], row["V_hi"] = _boot_ci(V, rng=rng, n_boot=n_boot)
        row["V_proposal_lo"], row["V_proposal_hi"] = _boot_ci(Vp, rng=rng, n_boot=n_boot)
        if np.isfinite(V).any() and row["V_proposal"] > 0:
            row["lift"] = row["V"] / row["V_proposal"]
            idx = rng.integers(0, n, size=(n_boot, n))
            num, den = np.nanmean(V[idx], axis=1), np.mean(Vp[idx], axis=1)
            ratio = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
            row["lift_lo"], row["lift_hi"] = (float(np.nanquantile(ratio, 0.025)),
                                              float(np.nanquantile(ratio, 0.975)))
        else:
            row["lift"] = row["lift_lo"] = row["lift_hi"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def h1_test(df: pd.DataFrame):
    """H1: V decreases with severity (pooled over sea states; aggregated cells)."""
    sel = df[np.isfinite(df["V"])]
    rho, p = spearmanr(sel["degradation"], sel["V"])
    return {"spearman": float(rho), "p": float(p), "n": int(len(sel)), "pass": rho < -0.5}


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
    """Pre-registered criterion: lift >= min_lift in at least half of the cells
    where the proposal is starved (V_proposal < headroom). Also reports how
    many of those cells have a CI lower bound above 1 (lift significant)."""
    sel = df[(df["V_proposal"] < headroom) & np.isfinite(df["lift"])]
    if len(sel) == 0:
        return {"n_cells": 0, "frac_pass": np.nan, "pass": False}
    frac = float((sel["lift"] >= min_lift).mean())
    out = {"n_cells": int(len(sel)), "frac_pass": frac, "median_lift": float(sel["lift"].median()),
           "pass": frac >= 0.5}
    if "lift_lo" in sel:
        out["n_ci_above_1"] = int((sel["lift_lo"] > 1.0).sum())
    return out
