"""Harbor geometry and spike scenario.

All coordinates ported VERBATIM from ``WAMV_MPC::initializeHarborZones()``
(vrx_control/src/wamv_mpc.cpp, ros2_underactuate branch). Frame: the Gazebo
world/odometry frame used by the ICRA experiments (harbor boundary at
x = -570; the vessel approaches from x > -570).

The spike scenario targets Zone 2, reached through the channel between
Dock 1 and Dock 2. From ``SPIKE_START_POSE`` at least two homotopy classes
exist: through the inter-dock channel, or around the outside of either dock.
``scripts/00_precheck_geometry.py`` verifies this with side-seeded B1 runs
before any training (gate G4 prerequisite).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class Zone:
    id: int
    name: str
    vertices: tuple            # ((x, y), ...) CCW or CW; shapely handles either
    
    @property
    def polygon(self) -> Polygon:
        return Polygon(self.vertices)

    @property
    def center(self) -> np.ndarray:
        return np.mean(np.asarray(self.vertices, dtype=float), axis=0)

    def halfspaces(self):
        """Return (A, b) with A p <= b describing the convex zone polygon."""
        return _poly_halfspaces(np.asarray(self.vertices, dtype=float))


def _poly_halfspaces(V: np.ndarray):
    """Halfspace form of a convex polygon given vertices (either orientation)."""
    c = V.mean(axis=0)
    A_rows, b_rows = [], []
    n = len(V)
    for i in range(n):
        p, q = V[i], V[(i + 1) % n]
        e = q - p
        nrm = np.array([e[1], -e[0]])          # one of the two normals
        if nrm @ (c - p) > 0:                  # make it outward: A p <= b holds inside
            nrm = -nrm
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm = nrm / nn
        A_rows.append(nrm)
        b_rows.append(nrm @ p)
    return np.asarray(A_rows), np.asarray(b_rows)


# ---- Zones (wamv_mpc.cpp lines ~1269-1287) --------------------------------
ZONES = (
    Zone(0, "Zone 1", ((-580.0, 258.0), (-572.0, 241.0), (-600.0, 236.0), (-600.0, 248.0))),
    Zone(1, "Zone 2", ((-570.0, 223.0), (-568.0, 209.0), (-595.0, 208.0), (-595.0, 220.0))),
    Zone(2, "Zone 3", ((-568.0, 192.0), (-593.0, 191.0), (-593.0, 183.0), (-579.0, 184.0))),
)

# ---- Dock obstacle polygons (wamv_mpc.cpp lines ~1293-1305) ---------------
DOCK_VERTICES = (
    ((-572.0, 241.0), (-570.0, 223.0), (-595.0, 220.0), (-600.0, 236.0)),   # Dock 1
    ((-568.0, 209.0), (-568.0, 192.0), (-593.0, 191.0), (-595.0, 208.0)),   # Dock 2
)
DOCKS = tuple(Polygon(v) for v in DOCK_VERTICES)

HARBOR_BOUNDARY_X = -570.0     # wamv_mpc.cpp fastPlanning()

# ---- Spike scenario --------------------------------------------------------
# Start east of the harbor line, offset north of the straight line to Zone 2 so
# the baseline clips Dock 1's eastern corner; detours on both sides are free.
SPIKE_START_POSE = np.array([-528.0, 232.0, np.pi])   # x, y, psi (facing -x)
SPIKE_ZONE_ID = 1                                     # Zone 2

# ---- Detour polylines (interior waypoints only; start/goal are appended) ----
# Zone 2 is the channel BETWEEN Dock 1 and Dock 2, so the only alternatives to
# the direct eastern entry go around the outside of a dock and enter the
# channel from its WEST end (x < -595), exactly what the precheck B1 solution
# does (runs/precheck.png). These are the SINGLE source of detour geometry for
# the proposal mixture (data/proposal.py) and the B1 seed set
# (baselines/restarts.py), so both explore the same homotopy classes.
DETOUR_POLYLINES = {
    "north": ((-560.0, 244.0), (-585.0, 250.0), (-605.0, 238.0), (-603.0, 222.0), (-597.0, 216.0)),
    "south": ((-556.0, 192.0), (-580.0, 183.0), (-604.0, 186.0), (-605.0, 206.0), (-597.0, 213.0)),
}


def detour_waypoints(name: str, x0: np.ndarray, zone: Zone, n_pts: int) -> np.ndarray:
    """Arc-length-uniform resampling of start -> polyline -> zone centroid."""
    pts = np.vstack([np.asarray(x0[:2], float), np.asarray(DETOUR_POLYLINES[name], float),
                     zone.center])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    st = np.linspace(0.0, s[-1], n_pts)
    return np.stack([np.interp(st, s, pts[:, j]) for j in range(2)], axis=1)


def detour_length(name: str, x0: np.ndarray, zone: Zone) -> float:
    pts = np.vstack([np.asarray(x0[:2], float), np.asarray(DETOUR_POLYLINES[name], float),
                     zone.center])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def feasible_zones(x0: np.ndarray, h1: float, h2: float, w_bar: float,
                   v_nom: float = 1.0, T_max: float = 150.0) -> list[Zone]:
    """Spike approximation of the reachability pre-filter p_g in R_Tmax(x0, H).

    The repo exposes no membership function (Phase 0 finding: port required),
    so the spike uses a conservative travel-time bound with an effective speed
    scaled by remaining authority plus an environmental-assistance floor.
    Refine in the full build.
    """
    authority = 0.5 * (h1 + h2)
    v_eff = v_nom * float(np.clip(authority + min(0.3, 0.002 * w_bar), 0.2, 1.0))
    p0 = np.asarray(x0[:2], dtype=float)
    out = []
    for z in ZONES:
        if np.linalg.norm(z.center - p0) / v_eff <= T_max:
            out.append(z)
    return out
