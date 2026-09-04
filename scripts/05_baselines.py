import _common  # noqa: F401
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import SPIKE_START_POSE, ZONES
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.score.obstacles import DockField
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.certify.surrogate import alpha_bar_policy
from manifold_recovery.baselines.restarts import run_b1, PlannerBank
from manifold_recovery.baselines.cem_mixture import run_b2
from manifold_recovery.baselines.rejection import run_b4

cfg = load(_common.ROOT / "configs/spike.yaml")
rtp = RTP(cfg.trajectory)
field = DockField()
rng = np.random.default_rng(3)
sampler = SyntheticSampler()
T_h, dt = rtp.horizon()
h1 = 0.25
w, sig = sampler.sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
w = w[:cfg.trajectory.N]
ab = float(alpha_bar_policy(sig, h1, 1.0, cfg))
x0 = SPIKE_START_POSE
x0f = np.array([x0[0], x0[1], x0[2], 0, 0, 0])
lines = [f"# Baselines @ h1={h1}, sea state {cfg.data.sea_state}, canonical start", ""]
b1 = run_b1(x0f, np.array([h1, 1.0]), ab, T_h, w, PlannerBank(cfg))
lines.append(f"B1 restarts: best FEASIBLE effort {b1['best']['cost'] if b1['best'] else None}, "
             f"feasible {b1['n_feasible']}/{b1['n_seeds']} (converged {b1['n_converged']}), "
             f"classes {b1['classes']}, wall {b1['wall_time']:.1f}s")
for at in b1["attempts"]:
    lines.append(f"    {ZONES[at['zone']].name} {at['seed']:9s} {at['status']:28s} "
                 f"slack {at['slack_total']:8.3f} effort {at['cost']:10.1f} sig {at['signature']}")
b2 = run_b2(x0, [h1, 1.0], lambda n: np.broadcast_to(w, (n,) + w.shape), rtp, field, cfg, rng)
lines.append(f"B2 CEM: best R {b2['best_R']:.1f}, classes {len(b2['classes'])} {b2['classes']}, "
             f"wall {b2['wall_time']:.1f}s")
b4 = run_b4(x0, h1, 1.0, w, sig, rtp, field, cfg, rng, wall_budget_s=5.0)
lines.append(f"B4 rejection (5s): acceptance {b4['acceptance']:.3f} over "
             f"{b4['n_tried']} samples, classes {len(b4['classes'])} {b4['classes']}")
try:
    from manifold_recovery.baselines.per_instance_lsmo import run_b3
    b3 = run_b3(x0, h1, lambda n: np.broadcast_to(w, (n,) + w.shape), rtp, field, cfg, rng)
    lines.append(f"B3 per-instance LSMO: wall {b3['wall_time']:.1f}s (n_train {b3['n_train']})")
except ImportError:
    lines.append("B3 per-instance LSMO: skipped (torch unavailable)")
out = _common.ROOT / "runs/baselines.md"
out.write_text("\n".join(lines))
print("\n".join(lines))