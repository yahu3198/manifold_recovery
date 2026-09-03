"""Homotopy-style strategy classes via obstacle side signatures (spec 4.3).

For each dock: find the waypoint of closest approach k*, and take the sign of
the 2-D cross product between the path tangent there and the vector to the
dock centroid. The tuple over docks is the class signature; ``cluster``
returns the best-scoring representative per signature.
"""
from __future__ import annotations

import numpy as np

from ..scenario import DOCK_VERTICES


_CENTROIDS = np.array([np.mean(v, axis=0) for v in DOCK_VERTICES])


def side_signature(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """pos, vel (..., N, 2) -> integer signatures (..., n_docks)."""
    sigs = []
    for c in _CENTROIDS:
        d = np.linalg.norm(pos - c, axis=-1)
        kstar = d.argmin(axis=-1)
        idx = np.expand_dims(kstar, -1)
        p = np.take_along_axis(pos, idx[..., None], axis=-2).squeeze(-2)
        v = np.take_along_axis(vel, idx[..., None], axis=-2).squeeze(-2)
        rel = c - p
        cross = v[..., 0] * rel[..., 1] - v[..., 1] * rel[..., 0]
        sigs.append(np.sign(cross).astype(int))
    return np.stack(sigs, axis=-1)


def cluster(pos, vel, score) -> dict:
    sigs = side_signature(pos, vel)
    out = {}
    flat_sig = sigs.reshape(-1, sigs.shape[-1])
    flat_score = np.asarray(score).reshape(-1)
    for i, s in enumerate(map(tuple, flat_sig)):
        if s not in out or flat_score[i] > flat_score[out[s]]:
            out[s] = i
    return out
