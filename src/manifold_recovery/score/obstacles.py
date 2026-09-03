"""Dock geometry costs (spec 4.4, Osa Eq. 31-32 hinge).

Signed distance is computed against precomputed edge arrays (vectorised
point-segment distances) with an inside test via matplotlib Path; per-point
shapely calls are avoided for speed. Negative distance = inside a dock.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from ..scenario import DOCK_VERTICES


class DockField:
    def __init__(self, dock_vertices=DOCK_VERTICES):
        self.paths = [MplPath(np.asarray(v, float)) for v in dock_vertices]
        segs_a, segs_b = [], []
        for v in dock_vertices:
            V = np.asarray(v, float)
            for i in range(len(V)):
                segs_a.append(V[i])
                segs_b.append(V[(i + 1) % len(V)])
        self.A = np.asarray(segs_a)            # (E, 2)
        self.B = np.asarray(segs_b)            # (E, 2)
        self.AB = self.B - self.A
        self.AB2 = np.maximum((self.AB ** 2).sum(-1), 1e-12)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """pts (..., 2) -> signed distance to nearest dock boundary (..., )."""
        shp = pts.shape[:-1]
        P = pts.reshape(-1, 2)                                     # (M, 2)
        # point-segment distances, (M, E)
        AP = P[:, None, :] - self.A[None, :, :]
        t = np.clip((AP * self.AB[None]).sum(-1) / self.AB2[None], 0.0, 1.0)
        proj = self.A[None] + t[..., None] * self.AB[None]
        d = np.linalg.norm(P[:, None, :] - proj, axis=-1).min(axis=1)
        inside = np.zeros(len(P), dtype=bool)
        for path in self.paths:
            inside |= path.contains_points(P)
        d = np.where(inside, -d, d)
        return d.reshape(shp)


def hinge_cost(d: np.ndarray, eps_obs: float) -> np.ndarray:
    """Osa Eq. 32: 0 for d > eps; quadratic in the margin band; linear inside."""
    out = np.zeros_like(d)
    band = (d > 0) & (d < eps_obs)
    out[band] = (1.0 / (2.0 * eps_obs)) * (d[band] - eps_obs) ** 2
    inside = d <= 0
    out[inside] = -d[inside] + eps_obs / 2.0
    return out


def j_obs(field: DockField, pos: np.ndarray, vel: np.ndarray, eps_obs: float):
    """J_obs = sum_k cost(d_k) * ||pdot_k|| (spec 4.4). Returns (J, d)."""
    d = field.signed_distance(pos)
    speed = np.linalg.norm(vel, axis=-1)
    J = (hinge_cost(d, eps_obs) * speed).sum(axis=-1)
    return J, d
