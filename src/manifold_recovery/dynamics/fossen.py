"""Nominal 3-DOF model as batched NumPy functions.

Exactly mirrors ``external/wamv_reference.py`` (the deployed ACADOS model):
diagonal M, zero added mass, rigid-body Coriolis terms only, quadratic+linear
damping. ``tests/test_dynamics_consistency.py`` pins this file against a
CasADi function built from the vendored reference to 1e-10.

Convention: M nudot + C(nu) nu + D(nu) nu = tau,  with plant tau = B H u + w
(the full environmental force acts once; R1-corrected formulation).
State x = [x, y, psi, u, v, r]; nu = [u, v, r] (body frame).
"""
from __future__ import annotations

import numpy as np

from .params import VesselParams, PARAMS, B_matrix


def M(p: VesselParams = PARAMS) -> np.ndarray:
    return np.diag([p.m, p.m, p.Izz])


def Minv(p: VesselParams = PARAMS) -> np.ndarray:
    return np.diag([1.0 / p.m, 1.0 / p.m, 1.0 / p.Izz])


def coriolis_vec(nu: np.ndarray, p: VesselParams = PARAMS) -> np.ndarray:
    """C(nu) nu, batched over leading dims. From wamv.py: +m v r in surge RHS
    means C nu = [-m v r, +m u r, 0] on the LHS."""
    u, v, r = nu[..., 0], nu[..., 1], nu[..., 2]
    out = np.empty_like(nu)
    out[..., 0] = -p.m * v * r
    out[..., 1] = p.m * u * r
    out[..., 2] = 0.0
    return out


def damping_vec(nu: np.ndarray, p: VesselParams = PARAMS) -> np.ndarray:
    """D(nu) nu on the LHS. wamv.py has +xu*u + xuu*|u|*u on the RHS with
    negative coefficients, i.e. LHS D nu = -(xu + xuu|u|) u etc. (>0 opposing)."""
    u, v, r = nu[..., 0], nu[..., 1], nu[..., 2]
    out = np.empty_like(nu)
    out[..., 0] = -(p.xu + p.xuu * np.abs(u)) * u
    out[..., 1] = -(p.yv + p.yvv * np.abs(v)) * v
    out[..., 2] = -(p.nr + p.nrr * np.abs(r)) * r
    return out


def required_wrench(nu: np.ndarray, nudot: np.ndarray,
                    p: VesselParams = PARAMS) -> np.ndarray:
    """tau_req = M nudot + C(nu) nu + D(nu) nu.  NO environmental subtraction:
    the environment enters only through the feasible wrench set W (R1 fix)."""
    Mdiag = np.array([p.m, p.m, p.Izz])
    return nudot * Mdiag + coriolis_vec(nu, p) + damping_vec(nu, p)


def Rz(psi: np.ndarray) -> np.ndarray:
    """Batched planar rotation matrices, shape (..., 2, 2)."""
    c, s = np.cos(psi), np.sin(psi)
    out = np.empty(psi.shape + (2, 2))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def f_continuous(x: np.ndarray, u_thr: np.ndarray, w: np.ndarray,
                 h: np.ndarray, p: VesselParams = PARAMS) -> np.ndarray:
    """State derivative for the closed-loop rollout (certify/exact).

    x: (..., 6), u_thr: (..., 2) thruster commands, w: (..., 3) body-frame
    environmental wrench (full force acts on the plant), h: (..., 2) health.
    """
    psi = x[..., 2]
    nu = x[..., 3:6]
    B = B_matrix(p.l)
    tau = np.einsum("ij,...j->...i", B, h * u_thr) + w
    nudot = (tau - coriolis_vec(nu, p) - damping_vec(nu, p)) / np.array([p.m, p.m, p.Izz])
    c, s = np.cos(psi), np.sin(psi)
    out = np.empty_like(x)
    out[..., 0] = c * nu[..., 0] - s * nu[..., 1]
    out[..., 1] = s * nu[..., 0] + c * nu[..., 1]
    out[..., 2] = nu[..., 2]
    out[..., 3:6] = nudot
    return out


def rk4_step(x, u_thr, w, h, dt, p: VesselParams = PARAMS):
    k1 = f_continuous(x, u_thr, w, h, p)
    k2 = f_continuous(x + 0.5 * dt * k1, u_thr, w, h, p)
    k3 = f_continuous(x + 0.5 * dt * k2, u_thr, w, h, p)
    k4 = f_continuous(x + dt * k3, u_thr, w, h, p)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
