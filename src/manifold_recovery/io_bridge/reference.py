"""Convert a fine-tuned plan into the MPC's 8-column reference stream.

Layout consumed by WAMV_MPC::manifold_ref_cb (std_msgs/Float64MultiArray):
    data[0] = t0 (ROS seconds at which row 0 applies), data[1] = dt,
    data[2:] = rows of [x, y, psi_bounded, u, v, r, Tp, Ts] at dt.
"""
from __future__ import annotations

import numpy as np


def wrap_pi(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def plan_to_rows(plan, T_h: float, dt_out: float = 0.05, feedforward: bool = False,
                 hold_s: float = 30.0) -> np.ndarray:
    """plan: PlanResult with X (6, N+1) and U (2, N) over [0, T_h].
    Appends ``hold_s`` of the terminal pose at zero velocity so the MPC
    station-keeps inside the zone after arrival."""
    X = np.asarray(plan.X, float); U = np.asarray(plan.U, float)
    N = X.shape[1] - 1
    tk = np.linspace(0.0, T_h, N + 1)
    tu = np.linspace(0.0, T_h, N, endpoint=False)
    t = np.arange(0.0, T_h + dt_out, dt_out)
    rows = np.zeros((len(t), 8))
    for j in range(6):
        rows[:, j] = np.interp(t, tk, X[j])
    if feedforward:
        rows[:, 6] = np.interp(t, tu, U[0]); rows[:, 7] = np.interp(t, tu, U[1])
    rows[:, 2] = wrap_pi(rows[:, 2])
    if hold_s > 0:
        n_hold = int(hold_s / dt_out)
        hold = np.repeat(rows[-1:], n_hold, axis=0)
        hold[:, 3:] = 0.0
        rows = np.vstack([rows, hold])
    return rows


def rows_to_message_data(rows: np.ndarray, t0: float, dt_out: float = 0.05) -> list:
    return [float(t0), float(dt_out)] + np.asarray(rows, float).reshape(-1).tolist()
