"""Harbor geometry and spike scenario (rev 3).

Zones and docks are ported VERBATIM from ``WAMV_MPC::initializeHarborZones()``
(vrx_control/src/wamv_mpc.cpp, ros2_underactuate branch). Frame: the Gazebo
world/odometry frame used by the ICRA experiments.

Rev 3 adds the SHORELINE (confirmed by the author, 3 Sep 2026): the west
edges of the zones and docks are a quay wall, a northern mole runs from
(-600, 248) to the tip (-580, 258), and a southern mole from (-593, 183) to
the tip (-579, 184). Everything east of the mole tips is open water. The
basin is therefore enterable only through the ~74 m opening between the
tips, and Zone 2 is a dead-end slip reachable from the east only. Rev 1/2
"around-dock" routes entering Zone 2 from x < -595 were on land.

Rev 3 target: reach ANY point of ANY of the three zones (ICRA mission
criterion). The recovery classes are therefore (target zone, winding about
the docks): Zone 1 north of Dock 1, Zone 2 into the slip, Zone 3 south of
Dock 2. The start pose is SAMPLED on an arc east of the opening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
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

    @property
    def path(self) -> MplPath:
        return MplPath(np.asarray(self.vertices, float))

    def halfspaces(self):
        """Return (A, b) with A p <= b describing the convex zone polygon."""
        return _poly_halfspaces(np.asarray(self.vertices, dtype=float))

    def contains(self, pts: np.ndarray, tol: float = 0.05) -> np.ndarray:
        """Halfspace test with a small tolerance: boundary points count as
        inside (the effort-optimal OCP terminal sits exactly on the boundary)."""
        pts = np.asarray(pts, float)
        A, b = self.halfspaces()
        return np.all(pts @ A.T <= b + tol, axis=-1)

    def sample_inside(self, rng: np.random.Generator, n: int, margin: float = 2.0):
        """Uniform points inside the zone at least ``margin`` from its boundary
        (rejection sampling in the bounding box)."""
        V = np.asarray(self.vertices, float)
        lo, hi = V.min(0), V.max(0)
        A, b = self.halfspaces()
        out = np.empty((n, 2))
        got = 0
        while got < n:
            cand = rng.uniform(lo, hi, size=(4 * (n - got), 2))
            ok = np.all(cand @ A.T <= b - margin, axis=1)
            take = cand[ok][: n - got]
            out[got:got + len(take)] = take
            got += len(take)
        return out


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

# ---- Shoreline (rev 3) -----------------------------------------------------
SHORE_POLYLINE = ((-580.0, 258.0), (-600.0, 248.0), (-600.0, 236.0), (-595.0, 220.0),
                  (-595.0, 208.0), (-593.0, 191.0), (-593.0, 183.0), (-579.0, 184.0))
_XW = -700.0            # land extends indefinitely west; this is far enough
_MOLE_T = 2.0           # mole wall thickness, on the seaward-outside face
# Convex pieces (the OCP needs halfspace form); together they tile the land.
LAND_VERTICES = (
    ((-600.0, 248.0), (-600.0, 236.0), (_XW, 236.0), (_XW, 248.0)),            # quay, Zone 1 band
    ((-600.0, 236.0), (-595.0, 220.0), (_XW, 220.0), (_XW, 236.0)),            # quay, Dock 1 band
    ((-595.0, 220.0), (-595.0, 208.0), (_XW, 208.0), (_XW, 220.0)),            # quay, Zone 2 band
    ((-595.0, 208.0), (-593.0, 191.0), (_XW, 191.0), (_XW, 208.0)),            # quay, Dock 2 band
    ((-593.0, 191.0), (-593.0, 183.0), (_XW, 183.0), (_XW, 191.0)),            # quay, Zone 3 band
    ((-600.0, 248.0), (-580.0, 258.0), (-580.0, 258.0 + _MOLE_T), (-600.0, 248.0 + _MOLE_T)),  # N mole
    ((-593.0, 183.0), (-579.0, 184.0), (-579.0, 184.0 - _MOLE_T), (-593.0, 183.0 - _MOLE_T)),  # S mole
)
OBSTACLE_VERTICES = DOCK_VERTICES + LAND_VERTICES   # everything the hull must avoid
MOLE_TIPS = (np.array([-580.0, 258.0]), np.array([-579.0, 184.0]))
OPENING_CENTER = 0.5 * (MOLE_TIPS[0] + MOLE_TIPS[1])          # (-579.5, 221)

HARBOR_BOUNDARY_X = -570.0     # wamv_mpc.cpp fastPlanning()

# ---- Entry vias: one interior waypoint per zone, just seaward of the dock
# corner the direct route would otherwise clip. Used by the proposal bases and
# the B1 seeds (single source of route geometry).
ENTRY_VIAS = {
    0: (-566.0, 247.0),    # Zone 1: NE of Dock 1's NE corner (-572, 241)
    1: (-560.0, 216.0),    # Zone 2: mouth of the slip
    2: (-562.0, 186.0),    # Zone 3: SE of Dock 2's SE corner (-568, 192)
}


def route_waypoints(zone: Zone, x0: np.ndarray, p_g: np.ndarray, n_pts: int) -> np.ndarray:
    """Arc-length-uniform resampling of start -> entry via -> p_g."""
    pts = np.vstack([np.asarray(x0[:2], float), np.asarray(ENTRY_VIAS[zone.id], float),
                     np.asarray(p_g, float)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    st = np.linspace(0.0, s[-1], n_pts)
    return np.stack([np.interp(st, s, pts[:, j]) for j in range(2)], axis=1)


def zone_of(pts: np.ndarray) -> np.ndarray:
    """(..., 2) -> zone index containing each point, or -1."""
    pts = np.asarray(pts, float)
    out = np.full(pts.shape[:-1], -1, dtype=int)
    for z in ZONES:
        out = np.where((out < 0) & z.contains(pts), z.id, out)
    return out


def in_any_zone(pts: np.ndarray) -> np.ndarray:
    return zone_of(pts) >= 0


def dist_outside_zones(pts: np.ndarray) -> np.ndarray:
    """(..., 2) -> distance-like penalty to the nearest zone: 0 inside any zone,
    else the smallest over zones of the largest halfspace violation (a lower
    bound on the Euclidean distance for convex zones; exact near an edge)."""
    pts = np.asarray(pts, float)
    best = None
    for z in ZONES:
        A, b = z.halfspaces()
        viol = np.maximum(pts @ A.T - b, 0.0).max(axis=-1)
        best = viol if best is None else np.minimum(best, viol)
    return best


# ---- Start pose ------------------------------------------------------------
# Canonical start (maps, precheck, baselines): 120 m due east of the opening,
# facing it. Datasets SAMPLE the start on an arc via ``sample_start``.
SPIKE_START_POSE = np.array([OPENING_CENTER[0] + 120.0, OPENING_CENTER[1], np.pi])
SPIKE_ZONE_ID = 1          # kept for legacy callers; rev 3 targets any zone


def sample_start(rng: np.random.Generator, n: int, d_range=(100.0, 140.0),
                 bearing_deg=(-35.0, 35.0), heading_jitter_deg: float = 45.0) -> np.ndarray:
    """(n, 3) poses on an arc east of the opening, facing it +- jitter."""
    d = rng.uniform(*d_range, size=n)
    b = np.deg2rad(rng.uniform(*bearing_deg, size=n))
    p = OPENING_CENTER[None, :] + np.stack([d * np.cos(b), d * np.sin(b)], axis=1)
    to_h = np.arctan2(OPENING_CENTER[1] - p[:, 1], OPENING_CENTER[0] - p[:, 0])
    psi = to_h + np.deg2rad(rng.uniform(-heading_jitter_deg, heading_jitter_deg, size=n))
    return np.column_stack([p, psi])


def feasible_zones(x0: np.ndarray, h1: float, h2: float, w_bar: float,
                   v_nom: float = 1.0, T_max: float = 150.0) -> list[Zone]:
    """Spike approximation of the reachability pre-filter p_g in R_Tmax(x0, H)."""
    authority = 0.5 * (h1 + h2)
    v_eff = v_nom * float(np.clip(authority + min(0.3, 0.002 * w_bar), 0.2, 1.0))
    p0 = np.asarray(x0[:2], dtype=float)
    return [z for z in ZONES if np.linalg.norm(z.center - p0) / v_eff <= T_max]