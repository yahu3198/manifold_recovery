"""G4 prerequisite: verify >= 2 homotopy classes exist for the spike scenario
BEFORE any data generation, via side-seeded B1 restarts. Writes
runs/precheck.png and exits nonzero if fewer than two classes are found."""
import _common  # noqa: F401
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.planner.energy_ocp import EnergyOCP
from manifold_recovery.baselines.restarts import run_b1
from manifold_recovery.certify.surrogate import alpha_bar_policy
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.analysis.plots import plot_trajectories

cfg = load(_common.ROOT / "configs/spike.yaml")
zone = ZONES[SPIKE_ZONE_ID]
rtp = RTP(cfg.trajectory)
T_h, dt = rtp.horizon(SPIKE_START_POSE, zone)
rng = np.random.default_rng(0)
w, sig = SyntheticSampler().sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
h = np.array([0.9, 1.0])
ab = float(alpha_bar_policy(sig, h[0], h[1], cfg))
x0 = np.array([*SPIKE_START_POSE, 0.0, 0.0, 0.0])
x0[2] = SPIKE_START_POSE[2]
x0 = np.array([SPIKE_START_POSE[0], SPIKE_START_POSE[1], SPIKE_START_POSE[2], 0, 0, 0])
planner = EnergyOCP(zone, cfg)
res = run_b1(x0, zone, h, ab, T_h, w, planner)
print(f"B1: {len(res['solutions'])} converged, classes = {res['classes']}, "
      f"best cost = {res['best']['cost'] if res['best'] else None}, "
      f"wall = {res['wall_time']:.1f}s")
(_common.ROOT / "runs").mkdir(exist_ok=True)
plot_trajectories([s["xi"] for s in res["solutions"]],
                  [hash(s["signature"]) % 7 for s in res["solutions"]],
                  f"B1 side-seeded solutions (h1=0.9), classes={len(res['classes'])}",
                  _common.ROOT / "runs/precheck.png", x0=SPIKE_START_POSE)
raise SystemExit(0 if len(res["classes"]) >= 2 else 1)
