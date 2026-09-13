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


def decoded_to_rows(pos: np.ndarray, vel: np.ndarray, psi: np.ndarray, T_h: float,
                    dt_out: float = 0.05, hold_s: float = 30.0) -> np.ndarray:
    """rev 4.5: reference rows from a certified DECODED candidate (stage 1).
    pos, vel: (N, 2) world frame over [0, T_h]; psi: (N,) heading from the
    inversion (includes the sideslip the screen assumed). Body velocities are
    the world velocity rotated into that heading; r is the heading rate."""
    pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    psi_u = np.unwrap(np.asarray(psi, float))
    N = len(pos)
    tk = np.linspace(0.0, T_h, N)
    t = np.arange(0.0, T_h + dt_out, dt_out)
    x = np.interp(t, tk, pos[:, 0]); y = np.interp(t, tk, pos[:, 1])
    vx = np.interp(t, tk, vel[:, 0]); vy = np.interp(t, tk, vel[:, 1])
    ps = np.interp(t, tk, psi_u)
    u = vx * np.cos(ps) + vy * np.sin(ps)
    v = -vx * np.sin(ps) + vy * np.cos(ps)
    rr = np.gradient(ps, dt_out)
    rows = np.zeros((len(t), 8))
    rows[:, 0] = x; rows[:, 1] = y; rows[:, 2] = wrap_pi(ps)
    rows[:, 3] = u; rows[:, 4] = v; rows[:, 5] = rr
    if hold_s > 0:
        n_hold = int(hold_s / dt_out)
        hold = np.repeat(rows[-1:], n_hold, axis=0)
        hold[:, 3:] = 0.0
        rows = np.vstack([rows, hold])
    return rows


def rows_to_message_data(rows: np.ndarray, t0: float, dt_out: float = 0.05) -> list:
    return [float(t0), float(dt_out)] + np.asarray(rows, float).reshape(-1).tolist()
