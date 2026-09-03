"""Dataset generation (spec 4.5-4.6, 5.4) and loading.

Each sample: draw h1 ~ U(h1_range) excluding the holdout band, a force
realisation w_seg from the EnvForceSampler, and omega from the proposal;
score with ``score_batch``; then apply per-stratum score shaping

    f(R) = exp(a (R - R_med) / (R_max - R_med))   if R >= R_med, else 0

with strata = h1 deciles (the shaping baseline the baselines B2-B4 also use).
Writes one npz (data contract, design doc Section 4) plus a JSON sidecar
carrying the config hash, stratum table, and ESS per stratum.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..scenario import ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID
from ..traj.rtp import RTP
from ..score.obstacles import DockField
from ..score.score import score_batch
from ..features.condition import build_c
from .proposal import ProposalSampler


def _draw_h1(n: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    lo, hi = cfg.data.h1_range
    hlo, hhi = cfg.data.h1_holdout
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.uniform(lo, hi, size=2 * (n - filled))
        cand = cand[(cand < hlo) | (cand > hhi)]
        take = min(len(cand), n - filled)
        out[filled:filled + take] = cand[:take]
        filled += take
    return out


def shape_weights(R: np.ndarray, h1: np.ndarray, a: float, n_bins: int = 10):
    """Per-h1-decile shaping. Returns f, and the stratum table for the sidecar."""
    edges = np.quantile(h1, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    f = np.zeros_like(R)
    table = []
    for i in range(n_bins):
        m = (h1 > edges[i]) & (h1 <= edges[i + 1])
        if m.sum() < 2:
            continue
        Rm, Rmax = np.median(R[m]), R[m].max()
        span = max(Rmax - Rm, 1e-9)
        fi = np.exp(a * (R[m] - Rm) / span)
        fi[R[m] < Rm] = 0.0
        f[m] = fi
        ess = float(fi.sum() ** 2 / max((fi ** 2).sum(), 1e-12))
        table.append({"h1_lo": float(edges[i]), "h1_hi": float(edges[i + 1]),
                      "R_med": float(Rm), "R_max": float(Rmax),
                      "n": int(m.sum()), "ess": ess})
    return f, table


def generate(cfg: Config, sampler, out_path: str | Path,
             x0=SPIKE_START_POSE, zone_id: int = SPIKE_ZONE_ID,
             inversion_mode: str = "crab", verbose: bool = True):
    rng = np.random.default_rng(cfg.data.seed)
    rtp = RTP(cfg.trajectory)
    field = DockField()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng)
    zone = ZONES[zone_id]
    _, dt = rtp.horizon(x0, zone)
    T_h = dt * (cfg.trajectory.N - 1)

    n = cfg.data.n_samples
    Bw2 = 2 * cfg.trajectory.Bw
    omega = np.empty((n, Bw2))
    h1 = _draw_h1(n, cfg, rng)
    h2 = np.full(n, cfg.data.h2_fixed)
    Rv = np.empty(n)
    w_all = np.empty((n, cfg.trajectory.N, 3), dtype=np.float32)
    sig_all = np.empty(n)
    term_keys = ("J_obs", "J_smooth", "J_feas", "J_effort", "min_clear", "d2_max")
    terms_acc = {k: np.empty(n) for k in term_keys}

    t0 = time.time()
    for s in range(0, n, cfg.data.chunk):
        e = min(s + cfg.data.chunk, n)
        m = e - s
        om = prop.sample(m, rng)
        w = np.empty((m, cfg.trajectory.N, 3))
        for i in range(m):
            wi, sg = sampler.sample(cfg.data.sea_state, cfg.data.direction,
                                    T_h, dt, rng)
            w[i] = wi[:cfg.trajectory.N]
            sig_all[s + i] = sg
        h = np.stack([h1[s:e], h2[s:e]], axis=1)
        R, terms = score_batch(om, h, w, x0, zone, rtp, field, cfg,
                               inversion_mode=inversion_mode)
        omega[s:e] = om
        Rv[s:e] = R
        w_all[s:e] = w
        for k in term_keys:
            terms_acc[k][s:e] = terms[k]
        if verbose:
            print(f"  scored {e}/{n}  ({time.time()-t0:.1f}s)", flush=True)

    f, table = shape_weights(Rv, h1, cfg.model.a_shaping)
    c = h1[:, None].astype(float)                       # spike condition = [h1]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, omega=omega, c=c, h1=h1, h2=h2, g_id=np.full(n, zone_id),
        R=Rv, f_weight=f, w_seg=w_all, sigma_theta=sig_all,
        x0=np.asarray(x0, float), dt=dt,
        **{f"terms_{k}": v for k, v in terms_acc.items()},
    )
    ess_total = float(f.sum() ** 2 / max((f ** 2).sum(), 1e-12))
    sidecar = {
        "config_hash": cfg.hash(), "n_samples": n, "zone_id": zone_id,
        "inversion_mode": inversion_mode, "ess_total": ess_total,
        "strata": table, "dt": dt, "T_h": T_h,
        "holdout_h1": list(cfg.data.h1_holdout),
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    if verbose:
        print(f"dataset -> {out_path}  ESS_total={ess_total:.0f}")
    return out_path


class ManifoldDataset:
    """Thin torch Dataset over the npz (import torch lazily)."""

    def __init__(self, npz_path: str | Path):
        d = np.load(npz_path)
        self.omega = d["omega"].astype(np.float32)
        self.c = d["c"].astype(np.float32)
        self.f = d["f_weight"].astype(np.float32)
        self.R = d["R"].astype(np.float32)
        self.meta = json.loads(Path(npz_path).with_suffix(".json").read_text())

    def __len__(self):
        return len(self.omega)

    def __getitem__(self, i):
        return self.omega[i], self.c[i], self.f[i]
