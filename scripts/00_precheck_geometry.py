"""G4 prerequisite (rev 3): verify that B1 finds feasible routes into >= 2 of
the three zones from the canonical start with the shoreline and docks as
obstacles. Writes runs/precheck.png; exits nonzero otherwise."""
import _common  # noqa: F401
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import SPIKE_START_POSE, ZONES
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.baselines.restarts import run_b1, PlannerBank
from manifold_recovery.certify.surrogate import alpha_bar_policy
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.analysis.plots import plot_trajectories

cfg = load(_common.ROOT / "configs/spike.yaml")
rtp = RTP(cfg.trajectory)
T_h, dt = rtp.horizon()
rng = np.random.default_rng(0)
w, sig = SyntheticSampler().sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
h = np.array([0.9, 1.0])
ab = float(alpha_bar_policy(sig, h[0], h[1], cfg))
x0 = np.array([SPIKE_START_POSE[0], SPIKE_START_POSE[1], SPIKE_START_POSE[2], 0, 0, 0])
res = run_b1(x0, h, ab, T_h, w, PlannerBank(cfg), verbose=True)
zones_hit = sorted({s["zone"] for s in res["solutions"]})
print(f"B1: {res['n_feasible']}/{res['n_seeds']} feasible ({res['n_converged']} converged), "
      f"classes = {res['classes']}, zones reached = {[ZONES[z].name for z in zones_hit]}, "
      f"best effort = {res['best']['cost'] if res['best'] else None}, wall = {res['wall_time']:.1f}s")
(_common.ROOT / "runs").mkdir(exist_ok=True)
plot_trajectories([s["xi"] for s in res["solutions"]],
                  [s["zone"] for s in res["solutions"]],
                  f"B1 feasible solutions (h1=0.9), classes={len(res['classes'])}",
                  _common.ROOT / "runs/precheck.png", x0=SPIKE_START_POSE, cbar_label="zone")
raise SystemExit(0 if len(zones_hit) >= 2 else 1)