#!/usr/bin/env python3
"""
Figure 1 (motivation) geometry export.

Draws ONLY geometry: harbour, piers, zones, three recovery routes, the
single-reference baseline path and the vessel glyph. No text, no axes, no
legend. All labels, the fault marker, wind/wave arrows, the scale bar and the
Gazebo inset are placed in TikZ over this PDF.

The export is deliberately NOT tight-cropped: the figure size is derived from
the data extent so that the data-to-image mapping is exactly linear. A data
point (x, y) lands at normalised image coordinate

    u = (x - XMIN) / (XMAX - XMIN),   v = (y - YMIN) / (YMAX - YMIN)

with u, v in [0, 1] measured from the bottom-left corner of the PDF. The script
prints a table of anchors already converted, plus a paste-ready TikZ block.

Usage
-----
    python fig1_motivation.py                       # placeholder routes
    python fig1_motivation.py --routes routes.npz   # real screened candidates
    python fig1_motivation.py --routes routes.npz --baseline internal.npy

Replace load_routes() and load_baseline_path() with loaders for your own data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle

# --------------------------------------------------------------------------
# Frame and extent
# --------------------------------------------------------------------------
# Plot frame is (x, y) in metres, matching the draft Fig. 1 axes.
# Round numbers on purpose: they make the normalised mapping easy to check.
XMIN, XMAX = -610.0, -444.0
YMIN, YMAX = 178.0, 266.0

FIG_W_IN = 3.45  # IEEE single column
FIG_H_IN = FIG_W_IN * (YMAX - YMIN) / (XMAX - XMIN)

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

ROUTE_COLOURS = {"Z1": "#0072B2", "Z2": "#E69F00", "Z3": "#CC79A7"}

VESSEL_SCALE = 4.0  # glyph is drawn oversize; state this in the caption


# --------------------------------------------------------------------------
# Harbour geometry
# --------------------------------------------------------------------------
def harbour_geometry() -> dict:
    """Harbour polygons.

    Tries scenario.py first so the figure cannot drift from the code that the
    screen and the planner use. Falls back to the literal VRX harbour values.
    """
    try:
        import scenario  # noqa: F401

        if hasattr(scenario, "figure_geometry"):
            return scenario.figure_geometry()
        print("[note] scenario.py found but has no figure_geometry(); "
              "using built-in values. Check they still match.")
    except ImportError:
        pass

    def rect(x0, x1, y0, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    return {
        # quay wall / land to the west
        "land": [rect(-612.0, -600.0, 178.0, 266.0)],
        # finger piers
        "piers": {
            "D1": rect(-600.0, -568.0, 223.0, 241.0),
            "D2": rect(-600.0, -568.0, 191.0, 208.0),
        },
        # moles north and south, open water east of their tips
        "moles": {
            "north": rect(-600.0, -545.0, 258.0, 261.0),
            "south": rect(-600.0, -545.0, 182.0, 184.0),
        },
        # berthing zones
        "zones": {
            "Z1": rect(-595.0, -568.0, 241.0, 258.0),
            "Z2": rect(-595.0, -568.0, 208.0, 223.0),
            "Z3": rect(-595.0, -568.0, 184.0, 191.0),
        },
        # canonical start, heading due west
        "start": np.array([-459.5, 221.0]),
        "start_heading": np.pi,
        # pier corner the single-reference baseline runs into
        "contact": np.array([-569.0, 223.0]),
    }


# --------------------------------------------------------------------------
# Route data
# --------------------------------------------------------------------------
def catmull_rom(ctrl: np.ndarray, n: int = 240) -> np.ndarray:
    """Centripetal Catmull-Rom through the control points."""
    p = np.asarray(ctrl, dtype=float)
    p = np.vstack([p[0] + (p[0] - p[1]), p, p[-1] + (p[-1] - p[-2])])
    out = []
    segs = len(p) - 3
    for i in range(segs):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        t = np.linspace(0.0, 1.0, max(2, n // segs))[:, None]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    return np.vstack(out)


def load_routes(path: str | None) -> dict:
    """Three screened candidates, one per homotopy class.

    Expects an .npz with arrays named Z1, Z2, Z3, each (N, 2) in (x, y) metres.
    Export these from the z-sweep at h1 = 0.5, taking the best-scoring member
    of each class, so the curvature in the figure is real.
    """
    if path:
        d = np.load(path)
        routes = {k: np.asarray(d[k], dtype=float) for k in ("Z1", "Z2", "Z3")}
        print(f"[data] routes loaded from {path}")
        return routes

    print("[PLACEHOLDER] synthetic routes. Replace with real screened "
          "candidates before submission.")
    return {
        "Z1": catmull_rom(
            [[-459.5, 221.0], [-497, 228], [-532, 240],
             [-556, 250], [-573, 252.5], [-585, 249.5]]
        ),
        "Z2": catmull_rom(
            [[-459.5, 221.0], [-500, 220], [-538, 218],
             [-563, 216], [-585, 215.5]]
        ),
        "Z3": catmull_rom(
            [[-459.5, 221.0], [-498, 211], [-532, 199],
             [-557, 189.5], [-572, 187.2], [-585, 187.5]]
        ),
    }


def load_baseline_path(path: str | None) -> np.ndarray:
    """Single-reference planner path, ending in pier contact.

    Expects an (N, 2) .npy in (x, y) metres. Take it from an internal-arm bag
    in the aligned block, where the contact is logged rather than drawn.
    """
    if path:
        arr = np.asarray(np.load(path), dtype=float)
        print(f"[data] baseline path loaded from {path}")
        return arr

    print("[PLACEHOLDER] synthetic baseline path. Replace with a real "
          "internal-arm trajectory before submission.")
    return catmull_rom(
        [[-459.5, 221.0], [-500, 222.2], [-538, 223.6],
         [-560, 223.4], [-569.0, 223.0]]
    )


# --------------------------------------------------------------------------
# Vessel glyph
# --------------------------------------------------------------------------
def vessel_glyph(ax, centre: np.ndarray, heading: float, scale: float):
    """Simple twin-pontoon WAM-V seen from above. Returns thruster anchors."""
    hull_l, pont_w, sep = 4.9, 1.0, 2.05  # metres, WAM-V
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])

    def place(pts):
        return (np.asarray(pts) * scale) @ R.T + centre

    for side in (+1, -1):
        y0 = side * sep / 2.0
        body = np.array(
            [
                [hull_l / 2, y0 - pont_w / 2],
                [hull_l / 2 - 0.9, y0 + pont_w / 2],
                [-hull_l / 2, y0 + pont_w / 2],
                [-hull_l / 2, y0 - pont_w / 2],
            ]
        )
        ax.add_patch(
            MplPolygon(place(body), closed=True, facecolor=C_VESSEL,
                       edgecolor="none", zorder=7)
        )

    deck = np.array(
        [[0.9, -sep / 2], [0.9, sep / 2], [-1.4, sep / 2], [-1.4, -sep / 2]]
    )
    ax.add_patch(
        MplPolygon(place(deck), closed=True, facecolor=C_VESSEL, alpha=0.55,
                   edgecolor="none", zorder=7)
    )

    # thruster anchors at the stern of each pontoon.
    # With heading pi (due west), port is the -y side.
    port = place([[-hull_l / 2 + 0.2, -sep / 2]])[0]
    stbd = place([[-hull_l / 2 + 0.2, +sep / 2]])[0]
    return {"port_thruster": port, "stbd_thruster": stbd}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw(geom: dict, routes: dict, baseline: np.ndarray, out_pdf: Path):
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # axes fill the figure exactly
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_axis_off()
    # NOTE: no set_aspect(); FIG_H_IN already makes the scales equal, and
    # an aspect constraint could pad the axes and break the mapping.

    ax.add_patch(
        Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN,
                  facecolor=C_WATER, edgecolor="none", zorder=0)
    )

    for poly in geom["land"]:
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_LAND,
                                edgecolor=C_LAND_EDGE, lw=0.6, zorder=1))
    for poly in geom["moles"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_LAND,
                                edgecolor=C_LAND_EDGE, lw=0.6, zorder=1))
    for poly in geom["piers"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_PIER,
                                edgecolor=C_PIER_EDGE, lw=0.7,
                                hatch="///", zorder=2))
    for poly in geom["zones"].values():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_ZONE,
                                edgecolor=C_ZONE_EDGE, lw=0.7, zorder=2))

    ax.plot(baseline[:, 0], baseline[:, 1], color=C_BASELINE, lw=1.3,
            ls=(0, (4, 2.2)), solid_capstyle="round", zorder=4)

    for key in ("Z1", "Z2", "Z3"):
        r = routes[key]
        ax.plot(r[:, 0], r[:, 1], color=ROUTE_COLOURS[key], lw=1.8,
                solid_capstyle="round", zorder=5)
        ax.plot(r[-1, 0], r[-1, 1], marker="o", ms=3.4,
                color=ROUTE_COLOURS[key], zorder=6)

    thr = vessel_glyph(ax, geom["start"], geom["start_heading"], VESSEL_SCALE)

    fig.savefig(out_pdf, format="pdf", pad_inches=0.0, transparent=True)
    # bbox_inches deliberately omitted: 'tight' would break the mapping.
    plt.close(fig)
    return thr


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------
def to_norm(p) -> tuple[float, float]:
    p = np.asarray(p, dtype=float)
    return ((p[0] - XMIN) / (XMAX - XMIN), (p[1] - YMIN) / (YMAX - YMIN))


def centroid(poly) -> np.ndarray:
    return np.asarray(poly, dtype=float).mean(axis=0)


def collect_anchors(geom, routes, thr) -> dict:
    a = {
        "vessel": geom["start"],
        "port_thruster": thr["port_thruster"],
        "stbd_thruster": thr["stbd_thruster"],
        "contact": geom["contact"],
    }
    for k, poly in geom["zones"].items():
        a[f"label_{k}"] = centroid(poly)
    for k, poly in geom["piers"].items():
        a[f"label_{k}"] = centroid(poly)
    for k, r in routes.items():
        a[f"end_{k}"] = r[-1]
        a[f"mid_{k}"] = r[int(0.62 * len(r))]
    # empty water, north east: candidate home for the Gazebo inset
    a["inset_anchor"] = np.array([-468.0, 258.0])
    a["wind_origin"] = np.array([-478.0, 243.0])
    a["scalebar_left"] = np.array([-604.0, 181.0])
    a["scalebar_right"] = np.array([-589.0, 181.0])  # 15 m
    return a


def report(anchors: dict, out_json: Path):
    print("\n" + "=" * 66)
    print(f"extent   x [{XMIN}, {XMAX}]   y [{YMIN}, {YMAX}]")
    print(f"figure   {FIG_W_IN:.3f} in x {FIG_H_IN:.3f} in "
          f"({(XMAX - XMIN) / FIG_W_IN:.2f} m/in, equal scales)")
    print("=" * 66)
    print(f"{'anchor':<18}{'x (m)':>10}{'y (m)':>10}{'u':>9}{'v':>9}")
    print("-" * 66)
    payload = {}
    for name, p in anchors.items():
        u, v = to_norm(p)
        payload[name] = {"x": float(p[0]), "y": float(p[1]),
                         "u": round(u, 4), "v": round(v, 4)}
        print(f"{name:<18}{p[0]:>10.1f}{p[1]:>10.1f}{u:>9.4f}{v:>9.4f}")

    out_json.write_text(json.dumps(
        {"extent": {"xmin": XMIN, "xmax": XMAX,
                    "ymin": YMIN, "ymax": YMAX},
         "figure_in": {"w": FIG_W_IN, "h": FIG_H_IN},
         "anchors": payload}, indent=2))

    print("\n" + "-" * 66)
    print("paste into the TikZ scope over the image:")
    print("-" * 66)
    for name, p in anchors.items():
        u, v = to_norm(p)
        print(f"\\coordinate ({name.replace('_', '')}) at ({u:.4f},{v:.4f});")
    print("-" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=None,
                    help=".npz with Z1, Z2, Z3 arrays of (N, 2) in metres")
    ap.add_argument("--baseline", default=None,
                    help=".npy with an (N, 2) internal-arm path in metres")
    ap.add_argument("--out", default="fig1_geometry.pdf")
    args = ap.parse_args()

    out_pdf = Path(args.out)
    geom = harbour_geometry()
    routes = load_routes(args.routes)
    baseline = load_baseline_path(args.baseline)

    thr = draw(geom, routes, baseline, out_pdf)
    anchors = collect_anchors(geom, routes, thr)
    report(anchors, out_pdf.with_suffix(".anchors.json"))
    print(f"\nwrote {out_pdf} and {out_pdf.with_suffix('.anchors.json')}")


if __name__ == "__main__":
    main()
