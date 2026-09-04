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
from ..scenario import ZONES, sample_start
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


def shape_weights(R: np.ndarray, h1: np.ndarray, a: float, n_bins: int = 10,
                  mode: np.ndarray | None = None, per_mode: bool = True,
                  feasible: np.ndarray | None = None):
    """Score shaping f(R) per stratum. Returns f and the stratum table.

    Strata are (h1 decile x mode). Within a stratum: f = exp(a (R - R_med) /
    (R_max - R_med)) for R >= R_med, else 0. Rev 3: each mode's shaped mass
    inside a decile is rescaled to the mode's FEASIBLE share (``feasible`` is
    a per-sample bool; share = feasible count of the mode / feasible count of
    the decile). A zone unreachable at a severity therefore fades from the
    manifold instead of being forced to its sample share (rev 2 error). Falls
    back to sample share if no sample in the decile is feasible.
    """
    edges = np.quantile(h1, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    f = np.zeros_like(R)
    table = []
    if mode is None or not per_mode:
        mode = np.zeros(len(R), dtype=int)
    if feasible is None:
        feasible = np.ones(len(R), dtype=bool)
    modes = np.unique(mode)
    for i in range(n_bins):
        m_bin = (h1 > edges[i]) & (h1 <= edges[i + 1])
        if m_bin.sum() < 2:
            continue
        n_feas_bin = int(feasible[m_bin].sum())
        for k in modes:
            m = m_bin & (mode == k)
            if m.sum() < 2:
                continue
            Rm, Rmax = np.median(R[m]), R[m].max()
            span = max(Rmax - Rm, 1e-9)
            fi = np.exp(a * (R[m] - Rm) / span)
            fi[R[m] < Rm] = 0.0
            share = (feasible[m].sum() / n_feas_bin if n_feas_bin > 0
                     else m.sum() / m_bin.sum())
            if fi.sum() > 0:
                fi *= share * m_bin.sum() / fi.sum()
            f[m] = fi
            ess = float(fi.sum() ** 2 / max((fi ** 2).sum(), 1e-12))
            table.append({"h1_lo": float(edges[i]), "h1_hi": float(edges[i + 1]),
                          "mode": int(k), "R_med": float(Rm), "R_max": float(Rmax),
                          "n": int(m.sum()), "share": float(share), "ess": ess})
    return f, table


def generate(cfg: Config, sampler, out_path: str | Path,
             inversion_mode: str = "crab", verbose: bool = True):
    rng = np.random.default_rng(cfg.data.seed)
    rtp = RTP(cfg.trajectory)
    field = DockField()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, mix=cfg.data.prop_mix)
    T_h, dt = rtp.horizon()

    n = cfg.data.n_samples
    omega = np.empty((n, rtp.dim))
    h1 = _draw_h1(n, cfg, rng)
    h2 = np.full(n, cfg.data.h2_fixed)
    x0_all = sample_start(rng, n, cfg.data.start_d_range, cfg.data.start_bearing_deg,
                          cfg.data.start_heading_jitter_deg)
    Rv = np.empty(n)
    w_all = np.empty((n, cfg.trajectory.N, 3), dtype=np.float32)
    sig_all = np.empty(n)
    mode_all = np.empty(n, dtype=np.int8)
    term_keys = ("J_obs", "J_smooth", "J_feas", "J_effort", "min_clear", "d2_max")
    terms_acc = {k: np.empty(n) for k in term_keys}

    t0 = time.time()
    for s in range(0, n, cfg.data.chunk):
        e = min(s + cfg.data.chunk, n)
        m = e - s
        om, mode = prop.sample_with_mode(x0_all[s:e], rng)
        mode_all[s:e] = mode
        w = np.empty((m, cfg.trajectory.N, 3))
        for i in range(m):
            wi, sg = sampler.sample(cfg.data.sea_state, cfg.data.direction,
                                    T_h, dt, rng)
            w[i] = wi[:cfg.trajectory.N]
            sig_all[s + i] = sg
        h = np.stack([h1[s:e], h2[s:e]], axis=1)
        R, terms = score_batch(om, h, w, x0_all[s:e], rtp, field, cfg,
                               inversion_mode=inversion_mode)
        omega[s:e] = om
        Rv[s:e] = R
        w_all[s:e] = w
        for k in term_keys:
            terms_acc[k][s:e] = terms[k]
        if verbose:
            print(f"  scored {e}/{n}  ({time.time()-t0:.1f}s)", flush=True)

    feasible = (terms_acc["min_clear"] >= 0.0) & (terms_acc["d2_max"] <= cfg.data.share_d2)
    f, table = shape_weights(Rv, h1, cfg.model.a_shaping, mode=mode_all,
                             per_mode=cfg.data.shape_per_mode, feasible=feasible)
    c = build_c(h1, x0_all, spike=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, omega=omega, c=c, h1=h1, h2=h2, g_id=mode_all.astype(int),
        R=Rv, f_weight=f, w_seg=w_all, sigma_theta=sig_all,
        prop_mode=mode_all, x0=x0_all, dt=dt, feasible=feasible,
        **{f"terms_{k}": v for k, v in terms_acc.items()},
    )
    ess_total = float(f.sum() ** 2 / max((f ** 2).sum(), 1e-12))
    mode_table = []
    for z in ZONES:
        mk = mode_all == z.id
        mode_table.append({"mode": z.id, "name": z.name, "n": int(mk.sum()),
                           "n_kept": int((f[mk] > 0).sum()),
                           "feasible_frac": float(feasible[mk].mean()) if mk.any() else float("nan"),
                           "f_mass": float(f[mk].sum() / max(f.sum(), 1e-12)),
                           "R_med": float(np.median(Rv[mk])) if mk.any() else float("nan")})
    sidecar = {
        "config_hash": cfg.hash(), "n_samples": n, "target": "any_zone",
        "inversion_mode": inversion_mode, "ess_total": ess_total,
        "strata": table, "dt": dt, "T_h": T_h,
        "holdout_h1": list(cfg.data.h1_holdout),
        "prop_mix": list(cfg.data.prop_mix), "prop_modes": mode_table,
        "horizon_mode": cfg.trajectory.horizon_mode, "dim_c": int(c.shape[1]),
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    if verbose:
        print(f"dataset -> {out_path}  ESS_total={ess_total:.0f}")
        for row in mode_table:
            print(f"  {row['name']}: n={row['n']} kept={row['n_kept']} "
                  f"feasible={row['feasible_frac']:.2f} f_mass={row['f_mass']:.2f} "
                  f"R_med={row['R_med']:.0f}")
    return out_path


class ManifoldDataset:
    """Thin torch Dataset over the npz (import torch lazily)."""

    def __init__(self, npz_path: str | Path):
        d = np.load(npz_path)
        self.omega = d["omega"].astype(np.float32)
        self.c = d["c"].astype(np.float32)
        self.f = d["f_weight"].astype(np.float32)
        self.R = d["R"].astype(np.float32)
        self.x0 = d["x0"]
        self.meta = json.loads(Path(npz_path).with_suffix(".json").read_text())

    def __len__(self):
        return len(self.omega)

    def __getitem__(self, i):
        return self.omega[i], self.c[i], self.f[i]