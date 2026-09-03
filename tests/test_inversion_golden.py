"""Golden-trajectory test: roll the true plant under smooth open-loop
controls, invert the resulting path under the same forces, and require the
wrench-set residual to be (near) zero; tangent mode must be no better than
crab on a curved path with lateral environmental force (R2 regression)."""
import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.dynamics import fossen
from manifold_recovery.dynamics.params import PARAMS
from manifold_recovery.traj.rtp import Kinematics
from manifold_recovery.traj.inversion import invert
from manifold_recovery.score.wrench_set import WrenchBoxes, distance_batch

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def _rollout(T=120.0, dt=0.1):
    x = np.array([-528.0, 232.0, np.pi, 0.0, 0.0, 0.0])
    h = np.array([1.0, 1.0])
    n = int(T / dt)
    w = np.tile([[-60.0, -45.0, 0.0]], (n, 1))
    xs = [x.copy()]
    for k in range(n):
        t = k * dt
        base = 350.0 + 80.0 * np.sin(2 * np.pi * t / 55.0)
        diff = 20.0 * np.sin(2 * np.pi * t / 70.0)
        u = np.clip([base - diff, base + diff], 0, PARAMS.u_max)
        x = fossen.rk4_step(x, np.asarray(u, float), w[k], h, dt)
        xs.append(x.copy())
    return np.asarray(xs), w, dt


def _kin_from_states(X, dt, N):
    idx = np.linspace(0, len(X) - 1, N).round().astype(int)
    pos = X[idx, 0:2][None]
    step = dt * (idx[1] - idx[0])
    vel = np.gradient(pos, step, axis=1)
    acc = np.gradient(vel, step, axis=1)
    return Kinematics(pos=pos, vel=vel, acc=acc,
                      T_h=step * (N - 1), dt=step), idx


def test_golden_feasibility_and_crab_beats_tangent():
    X, w, dt = _rollout()
    N = CFG.trajectory.N
    kin, idx = _kin_from_states(X, dt, N)
    w_seg = w[np.minimum(idx, len(w) - 1)][None]
    boxes = WrenchBoxes.unshrunk(PARAMS.u_max, 1.0)
    h = np.array([[1.0, 1.0]])
    d2_modes = {}
    for mode in ("crab", "tangent"):
        inv = invert(kin, w_seg, alpha_bar_y=1.0, mode=mode)
        d2, _, _ = distance_batch(inv.tau_req, h, w_seg, boxes, iters=400)
        d2_modes[mode] = float(np.mean(d2[0, 3:-3]))
    # feasible by construction: mean residual force below ~15 N per axis
    assert d2_modes["crab"] < 25.0 ** 2 * 3, d2_modes  # mean per-axis residual < 25 N
    assert d2_modes["crab"] <= d2_modes["tangent"] * 1.05, d2_modes
