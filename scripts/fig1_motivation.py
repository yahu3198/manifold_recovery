#!/usr/bin/env python3
"""
Figure 1 (motivation) geometry export, rev 2.

Draws ONLY geometry: harbour, piers, zones, the three recovery routes, the
single-reference baseline path and the vessel glyph. No text, no axes, no
legend, no colour bar. All labels, the fault marker, the wind and wave arrows,
the scale bar, the contact marker and the Gazebo inset are placed in TikZ over
this PDF.

Changes from rev 1
------------------
* Harbour geometry is IMPORTED from ``manifold_recovery.scenario``. Rev 1
  hardcoded axis-aligned rectangles; the real zones and docks are irregular
  quadrilaterals in an angled basin, and Zone 3 is an ~8 m slot. Land pieces
  extend west to x = -700 and are clipped to the frame here.
* Routes and the baseline path must be real data. Synthetic curves are only
  drawn with an explicit --allow-placeholder, so a placeholder cannot reach a
  submission by accident.
* The port thruster anchor follows Fossen's convention (body -y is port), so
  at psi = pi it lands on the +y (upper) side of the glyph in this frame.
  Rev 1 put it on the -y side, which would have attached the red cross to the
  starboard thruster.
* Checks are printed: terminal zone of each route, minimum clearance from the
  obstacle field, pairwise route separation converted to millimetres at the
  printed column width, and the baseline contact distance.

The export is deliberately NOT tight-cropped: the figure size is derived from
the data extent so that the data-to-image mapping is exactly linear. A data
point (x, y) lands at normalised image coordinate

    u = (x - XMIN) / (XMAX - XMIN),   v = (y - YMIN) / (YMAX - YMIN)

with u, v in [0, 1] measured from the bottom-left corner of the PDF.

Usage
-----
    python fig1_motivation.py --routes fig1_data/routes.npz \
                              --baseline fig1_data/baseline.npy \
                              --out fig1_geometry.pdf --debug-png

    python fig1_motivation.py --routes fig1_data/routes.npz --no-baseline
    python fig1_motivation.py --allow-placeholder          # layout work only

Produce the inputs with ``export_fig1_data.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle


# --------------------------------------------------------------------------
# Package bootstrap: works from scripts/ or from the repository root
# --------------------------------------------------------------------------
def _bootstrap():
    try:
        import manifold_recovery  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    for parent in Path(__file__).resolve().parents:
        cand = parent / "src"
        if (cand / "manifold_recovery").is_dir():
            sys.path.insert(0, str(cand))
            return
    raise ModuleNotFoundError(
        "manifold_recovery not importable; run from the repo or pip install -e ."
    )


_bootstrap()

from manifold_recovery.scenario import (  # noqa: E402
    ZONES, DOCK_VERTICES, LAND_VERTICES, MOLE_TIPS, OPENING_CENTER,
    SPIKE_START_POSE, zone_of,
)

# --------------------------------------------------------------------------
# Frame and extent
# --------------------------------------------------------------------------
# Plot frame is (x, y) in metres in the Gazebo world / odometry frame used by
# scenario.py and by the ICRA experiments. Round numbers on purpose: they make
# the normalised mapping easy to check by hand.
XMIN, XMAX = -610.0, -444.0
YMIN, YMAX = 178.0, 266.0

FIG_W_IN = 3.45  # IEEE single column
FIG_H_IN = FIG_W_IN * (YMAX - YMIN) / (XMAX - XMIN)
M_PER_IN = (XMAX - XMIN) / FIG_W_IN
M_PER_MM = M_PER_IN / 25.4

# --------------------------------------------------------------------------
# Palette (Okabe-Ito for the routes; red is reserved for the TikZ fault marker)
# --------------------------------------------------------------------------
C_WATER = "#EAF0F4"
C_LAND = "#D9D2C8"
C_LAND_EDGE = "#A89F91"
C_PIER = "#C4BCAE"
C_PIER_EDGE = "#8C8375"
C_ZONE = "#CDE8D5"
C_ZONE_EDGE = "#6FA880"
C_VESSEL = "#3A3A3A"
C_BASELINE = "#6E6E6E"

ROUTE_KEYS = ("Z1", "Z2", "Z3")
ROUTE_ZONE_ID = {"Z1": 0, "Z2": 1, "Z3": 2}
ROUTE_COLOURS = {"Z1": "#0072B2", "Z2": "#E69F00", "Z3": "#CC79A7"}

LW_ROUTE = 1.8
LW_BASELINE = 1.3
VESSEL_SCALE = 4.0  # glyph is drawn oversize; state this in the caption

# TikZ-side anchors that are pure layout choices, in metres.
WIND_ORIGIN = np.array([-492.0, 250.0])      # open water, north of the routes
WAVE_ORIGIN = np.array([-492.0, 244.0])
INSET_CENTRE = np.array([-466.0, 255.0])     # empty water, north east corner
SCALEBAR_LEFT = np.array([-546.0, 180.5])    # open water, south of the moles
SCALEBAR_RIGHT = SCALEBAR_LEFT + np.array([15.0, 0.0])   # 15 m


# --------------------------------------------------------------------------
# Harbour geometry, imported (never retyped)
# --------------------------------------------------------------------------
def clip_to_frame(poly) -> list[np.ndarray]:
    """Intersect a polygon with the figure frame. Returns 0, 1 or more rings.

    Land pieces in scenario.py run west to x = -700, far outside the frame;
    drawing them unclipped is harmless on screen but inflates the PDF and
    makes the bounding box depend on geometry that is never seen.
    """
    from shapely.geometry import Polygon as ShPoly, box

    g = ShPoly(np.asarray(poly, float)).buffer(0).intersection(
        box(XMIN, YMIN, XMAX, YMAX))
    if g.is_empty:
        return []
    geoms = list(getattr(g, "geoms", [g]))
    return [np.asarray(gg.exterior.coords)[:-1] for gg in geoms
            if gg.geom_type == "Polygon"]


def land_union() -> list:
    """Land pieces unioned, so the quay bands do not show internal seams.

    scenario.py tiles the land with seven convex pieces (the OCP needs
    halfspace form). Drawing them individually leaves hairlines across the
    quay that read as geometry the reader has to interpret.
    """
    from shapely.geometry import Polygon as ShPoly, box
    from shapely.ops import unary_union

    u = unary_union([ShPoly(np.asarray(v, float)).buffer(0)
                     for v in LAND_VERTICES]).intersection(
                         box(XMIN, YMIN, XMAX, YMAX))
    return [g for g in getattr(u, "geoms", [u]) if g.geom_type == "Polygon"]


def shapely_patch(poly, **kw):
    """PathPatch for a shapely polygon, holes included."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)
        verts.append(c)
        codes.append([MplPath.MOVETO] + [MplPath.LINETO] * (len(c) - 2)
                     + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(np.vstack(verts), np.concatenate(codes)), **kw)


def harbour_geometry() -> dict:
    """All polygons from scenario.py, clipped to the frame."""
    land = land_union()
    piers = {f"D{i + 1}": np.asarray(v, float) for i, v in enumerate(DOCK_VERTICES)}
    zones = {f"Z{z.id + 1}": np.asarray(z.vertices, float) for z in ZONES}
    return {
        "land": land,
        "piers": piers,
        "zones": zones,
        "zone_objs": {f"Z{z.id + 1}": z for z in ZONES},
        "start": np.asarray(SPIKE_START_POSE[:2], float),
        "start_heading": float(SPIKE_START_POSE[2]),
        "mole_tips": [np.asarray(t, float) for t in MOLE_TIPS],
        "opening": np.asarray(OPENING_CENTER, float),
    }


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def catmull_rom(ctrl: np.ndarray, n: int = 240) -> np.ndarray:
    """Centripetal Catmull-Rom through control points (placeholders only)."""
    p = np.asarray(ctrl, dtype=float)
    p = np.vstack([p[0] + (p[0] - p[1]), p, p[-1] + (p[-1] - p[-2])])
    out = []
    segs = len(p) - 3
    for i in range(segs):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        t = np.linspace(0.0, 1.0, max(2, n // segs))[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def load_routes(path: str | None, allow_placeholder: bool) -> tuple[dict, dict]:
    """Three screened candidates, one per homotopy class, (N, 2) in metres.

    Written by ``export_fig1_data.py``; the sidecar routes.meta.json records
    the seed, the severity and which candidate index each route came from.
    """
    if path:
        d = np.load(path)
        missing = [k for k in ROUTE_KEYS if k not in d]
        if missing:
            raise SystemExit(f"{path} is missing arrays {missing}")
        routes = {k: np.asarray(d[k], float) for k in ROUTE_KEYS}
        meta_path = Path(path).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        print(f"[data] routes from {path}"
              + (f" (seed {meta.get('seed')}, h1 {meta.get('h1')})" if meta else ""))
        if meta and not meta.get("all_screened", True):
            print("[WARN] routes.meta.json records at least one UNSCREENED route; "
                  "the caption must not call it screened")
        return routes, meta
    if not allow_placeholder:
        raise SystemExit(
            "no --routes given. Export real screened candidates first:\n"
            "    python scripts/export_fig1_data.py --h1 0.5 --seed 1\n"
            "or pass --allow-placeholder to work on the layout only.")
    print("[PLACEHOLDER] synthetic routes. NOT for submission.")
    return {
        "Z1": catmull_rom([[-459.5, 221.0], [-497, 228], [-532, 240],
                           [-556, 250], [-573, 250.5], [-584, 245.0]]),
        "Z2": catmull_rom([[-459.5, 221.0], [-500, 220], [-538, 218],
                           [-563, 216], [-585, 214.5]]),
        "Z3": catmull_rom([[-459.5, 221.0], [-498, 211], [-532, 199],
                           [-557, 189.5], [-572, 187.2], [-584, 187.0]]),
    }, {"placeholder": True}


def load_baseline_path(path: str | None, allow_placeholder: bool,
                       stride: int) -> np.ndarray | None:
    """Single-reference planner path ending in pier contact, (N, 2) in metres.

    From an internal-arm bag in the aligned block, where the contact is logged
    rather than drawn. ``export_fig1_data.py --bag ...`` writes it.
    """
    if path:
        arr = np.asarray(np.load(path), float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise SystemExit(f"{path}: expected (N, 2), got {arr.shape}")
        if stride > 1:
            arr = np.vstack([arr[::stride], arr[-1]])
        print(f"[data] baseline path from {path} ({len(arr)} points)")
        return arr
    if not allow_placeholder:
        return None
    print("[PLACEHOLDER] synthetic baseline path. NOT for submission.")
    return catmull_rom([[-459.5, 221.0], [-500, 222.2], [-538, 223.6],
                        [-560, 223.4], [-570.5, 223.4]])


# --------------------------------------------------------------------------
# Vessel glyph
# --------------------------------------------------------------------------
def vessel_glyph(ax, centre: np.ndarray, heading: float, scale: float):
    """Twin-pontoon WAM-V seen from above. Returns the thruster anchors.

    Body frame follows Fossen: x forward, y to STARBOARD. The port pontoon is
    therefore at body y = -l/2, and at psi = pi it rotates onto the +y side of
    this frame. The returned anchors are used for the TikZ red cross, so this
    sign is the one thing in the script worth re-checking by eye.
    """
    hull_l, pont_w, sep = 4.9, 1.0, 2.05  # metres, WAM-V
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])

    def place(pts):
        return (np.asarray(pts, float) * scale) @ R.T + centre

    for side in (+1, -1):                      # body y of each pontoon centre
        y0 = side * sep / 2.0
        body = np.array([[hull_l / 2, y0 - pont_w / 2],
                         [hull_l / 2 - 0.9, y0 + pont_w / 2],
                         [-hull_l / 2, y0 + pont_w / 2],
                         [-hull_l / 2, y0 - pont_w / 2]])
        ax.add_patch(MplPolygon(place(body), closed=True, facecolor=C_VESSEL,
                                edgecolor="none", zorder=7))

    deck = np.array([[0.9, -sep / 2], [0.9, sep / 2],
                     [-1.4, sep / 2], [-1.4, -sep / 2]])
    ax.add_patch(MplPolygon(place(deck), closed=True, facecolor=C_VESSEL,
                            alpha=0.55, edgecolor="none", zorder=7))

    port = place([[-hull_l / 2 + 0.2, -sep / 2]])[0]   # body -y = port
    stbd = place([[-hull_l / 2 + 0.2, +sep / 2]])[0]
    return {"port_thruster": port, "stbd_thruster": stbd}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw(geom: dict, routes: dict, baseline: np.ndarray | None, out_pdf: Path):
    plt.rcParams["hatch.linewidth"] = 0.5
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # axes fill the figure exactly
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_axis_off()
    # NOTE: no set_aspect(); FIG_H_IN already makes the scales equal, and an
    # aspect constraint could pad the axes and break the u, v mapping.

    ax.add_patch(Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN,
                           facecolor=C_WATER, edgecolor="none", zorder=0))

    for poly in geom["land"]:
        ax.add_patch(shapely_patch(poly, facecolor=C_LAND,
                                   edgecolor=C_LAND_EDGE, lw=0.6, zorder=1))
    for poly in geom["piers"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_PIER,
                                edgecolor=C_PIER_EDGE, lw=0.7, hatch="///",
                                zorder=2))
    for poly in geom["zones"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_ZONE,
                                edgecolor=C_ZONE_EDGE, lw=0.7, zorder=3))

    if baseline is not None:
        ax.plot(baseline[:, 0], baseline[:, 1], color=C_BASELINE,
                lw=LW_BASELINE, ls=(0, (4, 2.2)), solid_capstyle="round",
                zorder=4)

    for key in ROUTE_KEYS:
        r = routes[key]
        ax.plot(r[:, 0], r[:, 1], color=ROUTE_COLOURS[key], lw=LW_ROUTE,
                solid_capstyle="round", zorder=5)
        ax.plot(r[-1, 0], r[-1, 1], marker="o", ms=3.4,
                color=ROUTE_COLOURS[key], zorder=6)

    thr = vessel_glyph(ax, geom["start"], geom["start_heading"], VESSEL_SCALE)

    fig.savefig(out_pdf, format="pdf", pad_inches=0.0, transparent=True)
    # bbox_inches deliberately omitted: 'tight' would break the mapping.
    plt.close(fig)
    return thr


def draw_debug_png(geom, routes, baseline, thr, anchors, out_png: Path):
    """Same frame with labels drawn in, for eyeballing at true size.

    This file never goes in the paper. It exists so that "is the red cross on
    the port thruster" and "are the three routes distinct at 3.45 in" can be
    answered before any TikZ is written.
    """
    plt.rcParams["hatch.linewidth"] = 0.5
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_axis_off()
    ax.add_patch(Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN,
                           facecolor=C_WATER, edgecolor="none", zorder=0))
    for poly in geom["land"]:
        ax.add_patch(shapely_patch(poly, facecolor=C_LAND,
                                   edgecolor=C_LAND_EDGE, lw=0.6, zorder=1))
    for poly in geom["piers"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_PIER,
                                edgecolor=C_PIER_EDGE, lw=0.7, hatch="///",
                                zorder=2))
    for poly in geom["zones"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_ZONE,
                                edgecolor=C_ZONE_EDGE, lw=0.7, zorder=3))
    if baseline is not None:
        ax.plot(baseline[:, 0], baseline[:, 1], color=C_BASELINE,
                lw=LW_BASELINE, ls=(0, (4, 2.2)), zorder=4)
        ax.plot(*baseline[-1], marker="x", ms=5, mew=1.4, color="#B3202C",
                zorder=8)
    for key in ROUTE_KEYS:
        r = routes[key]
        ax.plot(r[:, 0], r[:, 1], color=ROUTE_COLOURS[key], lw=LW_ROUTE, zorder=5)
        mid = anchors[f"mid_{key}"]
        ax.text(mid[0], mid[1] + 2.0, key, color=ROUTE_COLOURS[key],
                fontsize=6, ha="center", zorder=9)
    vessel_glyph(ax, geom["start"], geom["start_heading"], VESSEL_SCALE)
    ax.plot(*thr["port_thruster"], marker="x", ms=6, mew=1.6, color="#D7191C",
            zorder=9)
    ax.text(thr["port_thruster"][0] + 2, thr["port_thruster"][1] + 4,
            "port 95%", fontsize=5.5, color="#D7191C", ha="center", zorder=9)
    for k, poly in geom["zones"].items():
        c = poly.mean(axis=0)
        ax.text(c[0], c[1], k, fontsize=5.5, ha="center", va="center", zorder=9)
    for k, poly in geom["piers"].items():
        c = poly.mean(axis=0)
        ax.text(c[0], c[1], k, fontsize=5.5, ha="center", va="center",
                color="#4A4A4A", zorder=9)
    ax.plot([SCALEBAR_LEFT[0], SCALEBAR_RIGHT[0]],
            [SCALEBAR_LEFT[1], SCALEBAR_RIGHT[1]], color="#3A3A3A", lw=1.0,
            zorder=9)
    ax.text(0.5 * (SCALEBAR_LEFT[0] + SCALEBAR_RIGHT[0]), SCALEBAR_LEFT[1] + 1.5,
            "15 m", fontsize=5, ha="center", zorder=9)
    fig.savefig(out_png, format="png", dpi=600, pad_inches=0.0)
    plt.close(fig)


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------
def to_norm(p) -> tuple[float, float]:
    p = np.asarray(p, float)
    return ((p[0] - XMIN) / (XMAX - XMIN), (p[1] - YMIN) / (YMAX - YMIN))


def collect_anchors(geom, routes, baseline, thr) -> dict:
    a = {
        "vessel": geom["start"],
        "port_thruster": thr["port_thruster"],
        "stbd_thruster": thr["stbd_thruster"],
    }
    for k, poly in geom["zones"].items():
        a[f"label_{k}"] = poly.mean(axis=0)
    for k, poly in geom["piers"].items():
        a[f"label_{k}"] = poly.mean(axis=0)
    for k in ROUTE_KEYS:
        r = routes[k]
        a[f"end_{k}"] = r[-1]
        a[f"mid_{k}"] = r[int(0.62 * (len(r) - 1))]
    if baseline is not None:
        a["contact"] = baseline[-1]
        a["mid_baseline"] = baseline[int(0.62 * (len(baseline) - 1))]
    a["opening"] = geom["opening"]
    a["mole_tip_n"], a["mole_tip_s"] = geom["mole_tips"]
    a["wind_origin"] = WIND_ORIGIN
    a["wave_origin"] = WAVE_ORIGIN
    a["inset_centre"] = INSET_CENTRE
    a["scalebar_left"] = SCALEBAR_LEFT
    a["scalebar_right"] = SCALEBAR_RIGHT
    return a


# --------------------------------------------------------------------------
# Checks (section 10 of the handoff, run automatically)
# --------------------------------------------------------------------------
SEP_IGNORE_M = 40.0   # the three routes share the start; ignore the fan-out


def min_curve_distance(a: np.ndarray, b: np.ndarray,
                       start: np.ndarray | None = None) -> float:
    """Minimum distance between two curves, optionally ignoring the region
    within SEP_IGNORE_M of the shared start where they are meant to coincide."""
    if start is not None:
        a = a[np.linalg.norm(a - start, axis=1) > SEP_IGNORE_M]
        b = b[np.linalg.norm(b - start, axis=1) > SEP_IGNORE_M]
        if len(a) == 0 or len(b) == 0:
            return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min())


def run_checks(geom, routes, baseline, thr) -> list[str]:
    msgs, warn = [], []

    # 1. terminal zone of each route
    for k in ROUTE_KEYS:
        z = int(zone_of(routes[k][-1]))
        want = ROUTE_ZONE_ID[k]
        ok = "ok" if z == want else "MISMATCH"
        msgs.append(f"  {k}: terminal zone {z} (expected {want})  {ok}")
        if z != want:
            warn.append(f"{k} does not end in its zone")

    # 2. clearance from the obstacle field (docks and land)
    try:
        from manifold_recovery.score.obstacles import DockField
        field = DockField()
        for k in ROUTE_KEYS:
            c = float(field.signed_distance(routes[k]).min())
            msgs.append(f"  {k}: min clearance {c:6.2f} m")
            if c < 0:
                warn.append(f"{k} intersects an obstacle")
        if baseline is not None:
            cb = float(field.signed_distance(baseline).min())
            msgs.append(f"  baseline: min clearance {cb:6.2f} m "
                        f"(negative = contact, which is the point)")
            if cb > 1.0:
                warn.append("baseline never touches a pier; the contact "
                            "marker in the caption would be unsupported")
    except Exception as e:  # obstacles needs shapely; do not fail the export
        msgs.append(f"  [clearance check skipped: {e}]")

    # 3. route distinctness at true size
    msgs.append(f"  (separation measured beyond {SEP_IGNORE_M:.0f} m from the "
                f"shared start)")
    for i, k1 in enumerate(ROUTE_KEYS):
        for k2 in ROUTE_KEYS[i + 1:]:
            d = min_curve_distance(routes[k1], routes[k2], geom["start"])
            msgs.append(f"  {k1} vs {k2}: min separation {d:6.2f} m "
                        f"= {d / M_PER_MM:5.2f} mm at {FIG_W_IN} in")
            if not np.isnan(d) and d / M_PER_MM < 1.5:
                warn.append(f"{k1} and {k2} come within 1.5 mm at true size; "
                            f"they will not read as distinct")

    # 3b. is the southern slot visible at all
    z3 = geom["zones"]["Z3"]
    slot_m = float(z3[:, 1].max() - z3[:, 1].min())
    msgs.append(f"  Zone 3 slot spans {slot_m:.1f} m = {slot_m / M_PER_MM:.1f} mm "
                f"at {FIG_W_IN} in")
    if slot_m / M_PER_MM < 2.0:
        warn.append("Zone 3 is under 2 mm tall at true size; consider a "
                    "tighter frame or a wider figure")

    # 4. port side of the glyph
    side = "+y (upper)" if thr["port_thruster"][1] > geom["start"][1] else "-y (lower)"
    msgs.append(f"  port thruster is on the {side} side of the glyph "
                f"(expected +y at psi = pi)")
    if thr["port_thruster"][1] <= geom["start"][1]:
        warn.append("port thruster on the wrong side; the red cross would "
                    "land on the starboard thruster")

    # 5. anything drawn outside the frame is silently clipped
    def outside(arr):
        return int(np.sum((arr[:, 0] < XMIN) | (arr[:, 0] > XMAX) |
                          (arr[:, 1] < YMIN) | (arr[:, 1] > YMAX)))
    for k in ROUTE_KEYS:
        n = outside(routes[k])
        if n:
            warn.append(f"{k} has {n} points outside the frame (clipped)")
    if baseline is not None and outside(baseline):
        warn.append(f"baseline has {outside(baseline)} points outside the frame")

    return msgs, warn


def report(anchors: dict, out_json: Path, check_msgs, warnings_):
    print("\n" + "=" * 72)
    print(f"extent   x [{XMIN}, {XMAX}]   y [{YMIN}, {YMAX}]")
    print(f"figure   {FIG_W_IN:.3f} in x {FIG_H_IN:.3f} in "
          f"({M_PER_IN:.2f} m/in = {M_PER_MM:.2f} m/mm, equal scales)")
    print("=" * 72)
    print(f"{'anchor':<18}{'x (m)':>10}{'y (m)':>10}{'u':>9}{'v':>9}")
    print("-" * 72)
    payload = {}
    for name, p in anchors.items():
        u, v = to_norm(p)
        payload[name] = {"x": float(p[0]), "y": float(p[1]),
                         "u": round(u, 4), "v": round(v, 4)}
        print(f"{name:<18}{p[0]:>10.1f}{p[1]:>10.1f}{u:>9.4f}{v:>9.4f}")

    out_json.write_text(json.dumps(
        {"extent": {"xmin": XMIN, "xmax": XMAX, "ymin": YMIN, "ymax": YMAX},
         "figure_in": {"w": FIG_W_IN, "h": FIG_H_IN},
         "m_per_mm": M_PER_MM,
         "anchors": payload}, indent=2))

    print("\n" + "-" * 72)
    print("checks")
    print("-" * 72)
    print("\n".join(check_msgs))

    print("\n" + "-" * 72)
    print("paste into the TikZ scope over the image "
          "(labels go on mid_*, not end_*: the three ends stack at the left edge)")
    print("-" * 72)
    for name, p in anchors.items():
        u, v = to_norm(p)
        print(f"\\coordinate ({name.replace('_', '')}) at ({u:.4f},{v:.4f});")
    print("-" * 72)

    if warnings_:
        print("\nWARNINGS")
        for w in warnings_:
            print(f"  ! {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=None,
                    help=".npz with Z1, Z2, Z3 arrays of (N, 2) in metres")
    ap.add_argument("--baseline", default=None,
                    help=".npy with an (N, 2) internal-arm path in metres")
    ap.add_argument("--no-baseline", action="store_true",
                    help="section 11 fallback: routes only, no grey path")
    ap.add_argument("--baseline-stride", type=int, default=4,
                    help="decimate the odometry path (it is logged at 20 Hz)")
    ap.add_argument("--allow-placeholder", action="store_true",
                    help="draw synthetic routes for layout work; never submit")
    ap.add_argument("--debug-png", action="store_true",
                    help="also write an annotated PNG for eyeballing")
    ap.add_argument("--out", default="fig1_geometry.pdf")
    args = ap.parse_args()

    out_pdf = Path(args.out)
    geom = harbour_geometry()
    routes, _ = load_routes(args.routes, args.allow_placeholder)
    baseline = (None if args.no_baseline
                else load_baseline_path(args.baseline, args.allow_placeholder,
                                        args.baseline_stride))

    thr = draw(geom, routes, baseline, out_pdf)
    anchors = collect_anchors(geom, routes, baseline, thr)
    msgs, warns = run_checks(geom, routes, baseline, thr)
    report(anchors, out_pdf.with_suffix(".anchors.json"), msgs, warns)

    if args.debug_png:
        png = out_pdf.with_name(out_pdf.stem + "_debug.png")
        draw_debug_png(geom, routes, baseline, thr, anchors, png)
        print(f"\nwrote {png} (eyeball this at true size, then discard it)")

    print(f"\nwrote {out_pdf} and {out_pdf.with_suffix('.anchors.json')}")


if __name__ == "__main__":
    main()
