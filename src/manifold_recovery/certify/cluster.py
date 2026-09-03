"""Homotopy classes via per-obstacle winding signatures (spec 4.3; rev 2).

Rev 1 used the sign of the cross product between the path tangent and the
vector to each dock centroid at closest approach. For this harbor that is NOT
a homotopy invariant: the north and south around-dock detours both enter the
Zone 2 channel from its west end and receive the identical signature (+1, -1),
so cluster() merged them (spike run 1 finding).

Rev 2 uses, per dock, the total signed angle the path sweeps about the dock
centroid, minus the angle swept by the straight chord between the same two
endpoints, quantised to whole turns:

    sig_j = round( (sweep_path(c_j) - sweep_chord(c_j)) / 2 pi )

For paths sharing endpoints this difference is exactly the winding number of
the closed loop (path followed by reversed chord), a true homotopy invariant
in the plane minus the dock. Straight -> (0, 0); north detour -> (1, 0);
south detour -> (0, -1). Fully vectorised over leading batch dims.

``side_signature`` is kept as the public name so every caller (gates,
contraction maps, B1/B2/B4, pipeline) picks the fix up unchanged.
"""
from __future__ import annotations

import numpy as np

from ..scenario import DOCK_VERTICES


_CENTROIDS = np.array([np.mean(v, axis=0) for v in DOCK_VERTICES])


def _sweep(pos: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed angle swept about c along the polyline pos (..., N, 2) -> (...,)."""
    rel = pos - c
    ang = np.unwrap(np.arctan2(rel[..., 1], rel[..., 0]), axis=-1)
    return ang[..., -1] - ang[..., 0]


def winding_signature(pos: np.ndarray, n_chord: int = 64) -> np.ndarray:
    """pos (..., N, 2) -> integer winding signatures (..., n_docks)."""
    pos = np.asarray(pos, float)
    t = np.linspace(0.0, 1.0, n_chord)
    chord = pos[..., :1, :] + t[:, None] * (pos[..., -1:, :] - pos[..., :1, :])
    sigs = []
    for c in _CENTROIDS:
        dw = _sweep(pos, c) - _sweep(chord, c)
        sigs.append(np.rint(dw / (2.0 * np.pi)).astype(int))
    return np.stack(sigs, axis=-1)


def side_signature(pos: np.ndarray, vel: np.ndarray | None = None) -> np.ndarray:
    """Backward-compatible entry point; ``vel`` is accepted and ignored."""
    return winding_signature(pos)


def cluster(pos, vel, score) -> dict:
    """Best-scoring representative index per signature tuple."""
    sigs = side_signature(pos, vel)
    out = {}
    flat_sig = sigs.reshape(-1, sigs.shape[-1])
    flat_score = np.asarray(score).reshape(-1)
    for i, s in enumerate(map(tuple, flat_sig.tolist())):
        if s not in out or flat_score[i] > flat_score[out[s]]:
            out[s] = i
    return out