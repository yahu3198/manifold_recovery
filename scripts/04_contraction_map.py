"""Contraction maps V, V_proposal, lift, N_H over severity x sea state (rev 4.1).

Rev 4.1: one row per force draw, --n-env draws per cell (default 10), a finer
severity axis (default 50/60/70/80/90/95/100 %), --n-starts random starts from
the arc in addition to the canonical start, and 95 % bootstrap CIs on V,
V_proposal and lift. Heatmaps are drawn at the canonical start; the pooled
table (all starts) and the per-draw table are saved alongside.
"""
import _common  # noqa: F401
import argparse
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import SPIKE_START_POSE, sample_start
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.analysis.contraction import (compute_maps, aggregate, h1_test,
                                                    h2_test, h3_test, lift_test)
from manifold_recovery.analysis.plots import plot_V_heatmap

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--ckpt", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--proposal-only", action="store_true")
p.add_argument("--n-env", type=int, default=10, help="force draws per cell")
p.add_argument("--n-starts", type=int, default=3, help="random starts from the arc (0 = canonical only)")
p.add_argument("--severities", default="0.5,0.6,0.7,0.8,0.9,0.95,1.0")
p.add_argument("--K", type=int, default=100)
p.add_argument("--seed", type=int, default=2)
a = p.parse_args()
cfg = load(a.config)
rng = np.random.default_rng(a.seed)
sev = tuple(float(v) for v in a.severities.split(","))
decoder = None
if not a.proposal_only:
    from manifold_recovery.model.train import load as load_ckpt
    from manifold_recovery.model.decode import Decoder
    model, meta = load_ckpt(a.ckpt)
    decoder = Decoder(model, meta, cfg)

starts = np.vstack([SPIKE_START_POSE[None, :],
                    sample_start(rng, a.n_starts, cfg.data.start_d_range,
                                 cfg.data.start_bearing_deg, cfg.data.start_heading_jitter_deg)]
                   ) if a.n_starts > 0 else None
draws = compute_maps(decoder, cfg, SyntheticSampler(), rng, severities=sev,
                     K=a.K, n_env=a.n_env, starts=starts, verbose=True)
(_common.ROOT / "runs").mkdir(exist_ok=True)
draws.to_csv(_common.ROOT / "runs/maps_draws.csv", index=False)

canon = aggregate(draws[draws["start"] == 0])
pooled = aggregate(draws)                       # all starts and draws per cell
canon.to_csv(_common.ROOT / "runs/maps.csv", index=False)
pooled.to_csv(_common.ROOT / "runs/maps_pooled.csv", index=False)

plot_V_heatmap(canon, "V_proposal", _common.ROOT / "runs/heatmap_V_proposal.png",
               title="V_proposal over severity x sea state (canonical start)")
if not a.proposal_only:
    plot_V_heatmap(canon, "V", _common.ROOT / "runs/heatmap_V.png",
                   title="V over severity x sea state (canonical start)")
    plot_V_heatmap(canon, "lift", _common.ROOT / "runs/heatmap_lift.png",
                   title="lift = V / V_proposal (canonical start)")

cols = ["degradation", "sea_state", "n_draws", "V", "V_lo", "V_hi", "V_proposal",
        "V_proposal_lo", "V_proposal_hi", "lift", "lift_lo", "lift_hi", "N_H"]
print(f"\n# Canonical start ({a.n_env} draws per cell, 95% bootstrap CI)")
print(canon[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
if starts is not None and len(starts) > 1:
    print(f"\n# Pooled over {len(starts)} starts ({a.n_env * len(starts)} draws per cell)")
    print(pooled[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
if not a.proposal_only:
    print("\nH1 (canonical):", h1_test(canon))
    print("H1 (pooled):   ", h1_test(pooled))
    print("H2:", h2_test(canon))
    print("H3 (canonical, cells on the ICRA grid):", h3_test(canon))
    print("LIFT (pre-registered, canonical):", lift_test(canon))
    print("LIFT (pooled over starts):        ", lift_test(pooled))
