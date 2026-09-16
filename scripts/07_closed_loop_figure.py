#!/usr/bin/env python3
"""Assemble the 3x2 closed-loop figure from the Gazebo ghost screenshots.

Step 1, once per camera pose: calibrate pixel <-> world from the ghost hulls,
whose world poses are known exactly from <cell>.json.
    python scripts/07_closed_loop_figure.py --calibrate-ghosts \
        ~/usv_ws/experiments/figure_ghosts/d50_beneficial.png ~/usv_ws/experiments/figure_ghosts/d95_nominal.png
For each screenshot the hulls are prompted in order with their time stamp; click
the hull centre (mid-length, between the pontoons), right-click to skip one.
The affine is fitted over all clicked hulls, saved as calibration.json with its
residual, and calibration_check.png shows every JSON hull pose drawn on every
screenshot so the fit can be verified by eye. --calibrate (pier corners) and
--calibrate-from ("x y px py" lines) remain available.

Step 2: build the figure.
    python scripts/07_closed_loop_figure.py --dir ~/usv_ws/experiments/figure_ghosts

Reads <cell>.png and <cell>.json for the six cells and lays them out 2 x 3: rows
aligned / quartering, columns 50 / 80 / 95 % degradation. Crops every panel to
the same world extent, overlays the exact zone and pier polygons the scorer uses, marks
the terminal event of the manifold arm and of the internal planner (arrival,
pier collision, stop short), labels the panels, adds a scale bar and one legend,
and writes closed_loop.pdf, closed_loop.png and caption.txt.

The camera is nadir and north-up, so a plane at z=0 maps to the image by an
affine with negligible rotation; the fit checks that and warns otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from manifold_recovery.scenario import ZONES, DOCK_VERTICES, MOLE_TIPS, SHORE_POLYLINE  # noqa: E402

ROWS, COLS = ["beneficial", "nominal"], ["d50", "d80", "d95"]
CELLS = [(sev, d) for d in ROWS for sev in COLS]          # panel order (a)..(f), row-major
DIR_LABEL = {"beneficial": "aligned", "nominal": "quartering"}
SEV_LABEL = {"d50": "50 %", "d80": "80 %", "d95": "95 %"}

# named landmarks for calibration: (label, x, y). Pier corners first: they are
# the crispest features in the render.
LANDMARKS = [
    ("Dock 1 north-east corner", *DOCK_VERTICES[0][0]),
    ("Dock 1 south-east corner", *DOCK_VERTICES[0][1]),
    ("Dock 2 north-east corner", *DOCK_VERTICES[1][0]),
    ("Dock 2 south-east corner", *DOCK_VERTICES[1][1]),
    ("Dock 1 south-west corner", *DOCK_VERTICES[0][2]),
    ("Dock 2 south-west corner", *DOCK_VERTICES[1][2]),
    ("north mole tip", *MOLE_TIPS[0]),
    ("south mole tip", *MOLE_TIPS[1]),
]

C_TRACK, C_REF, C_REPLAN, C_INTERNAL = "#1a66f2", "#e61a1a", "#ff8c26", "#4d4d4d"


# ------------------------------------------------------------------ calibration
def fit_affine(world: np.ndarray, pix: np.ndarray):
    """pix = A @ world + b, least squares. Returns (A, b, residual_px)."""
    X = np.hstack([world, np.ones((len(world), 1))])
    M, *_ = np.linalg.lstsq(X, pix, rcond=None)          # (3, 2)
    A, b = M[:2].T, M[2]
    res = np.hypot(*(X @ M - pix).T)
    return A, b, res


def save_calibration(path: Path, world, pix, image_size):
    A, b, res = fit_affine(np.asarray(world, float), np.asarray(pix, float))
    sx, sy = np.hypot(*A[:, 0]), np.hypot(*A[:, 1])
    shear = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    print(f"scale {sx:.2f} px/m (x), {sy:.2f} px/m (y); rotation {shear:.2f} deg; "
          f"residual {res.mean():.1f} px mean, {res.max():.1f} px max ({res.max()/sx:.2f} m)")
    if abs(shear) > 1.5:
        print("WARNING: rotation above 1.5 deg; the camera is not north-up, the crop will be slightly skewed")
    cal = {"A": A.tolist(), "b": b.tolist(), "image_size": list(image_size),
           "world": np.asarray(world, float).tolist(), "pix": np.asarray(pix, float).tolist(),
           "residual_px": res.tolist()}
    path.write_text(json.dumps(cal, indent=1))
    print(f"saved {path}")
    return cal


def calibrate_interactive(png: Path, out: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    img = plt.imread(png)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img); ax.set_title("click the landmarks in order (right-click to skip one, Enter when done)")
    print("Landmarks (world x, y):")
    for i, (name, x, y) in enumerate(LANDMARKS):
        print(f"  {i + 1}. {name}  ({x:.0f}, {y:.0f})")
    plt.show(block=False)
    world, pix = [], []
    for name, x, y in LANDMARKS:
        ax.set_xlabel(f"click: {name} ({x:.0f}, {y:.0f})  |  right-click to skip"); fig.canvas.draw()
        pts = plt.ginput(1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)
        if not pts:
            print(f"  skipped {name}"); continue
        world.append((x, y)); pix.append(pts[0]); ax.plot(*pts[0], "r+", ms=12); fig.canvas.draw()
        print(f"  {name}: px ({pts[0][0]:.0f}, {pts[0][1]:.0f})")
    plt.close(fig)
    if len(world) < 3:
        sys.exit("need at least three landmarks")
    save_calibration(out, world, pix, (img.shape[1], img.shape[0]))


def calibrate_from_file(txt: Path, png: Path, out: Path):
    import matplotlib.pyplot as plt
    rows = [list(map(float, l.split())) for l in txt.read_text().splitlines() if l.strip() and not l.startswith("#")]
    world = [r[:2] for r in rows]; pix = [r[2:4] for r in rows]
    img = plt.imread(png)
    save_calibration(out, world, pix, (img.shape[1], img.shape[0]))


def calibrate_ghosts(pngs: list[Path], out: Path):
    import matplotlib.pyplot as plt
    world, pix, size = [], [], None
    for png in pngs:
        js = png.with_suffix(".json")
        if not js.exists():
            print(f"skip {png.name}: no {js.name}"); continue
        cell = json.loads(js.read_text()); img = plt.imread(png); size = (img.shape[1], img.shape[0])
        fig, ax = plt.subplots(figsize=(14, 10)); ax.imshow(img)
        ax.set_title(f"{cell['cell']}: click each hull centre in order (right-click to skip, close window to stop)")
        plt.show(block=False)
        for g in cell["ghosts"]:
            ax.set_xlabel(f"hull at t = {g['t_since_fault']:.0f} s  {g.get('tag', '')}  world ({g['x']:.1f}, {g['y']:.1f})")
            fig.canvas.draw()
            pts = plt.ginput(1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)
            if not pts:
                print(f"  skipped t={g['t_since_fault']:.0f}"); continue
            world.append((g["x"], g["y"])); pix.append(pts[0])
            ax.plot(*pts[0], "r+", ms=12); fig.canvas.draw()
        plt.close(fig)
    if len(world) < 3:
        sys.exit("need at least three hulls")
    cal = save_calibration(out, world, pix, size)
    write_check_image(out.parent, Cal(cal))


def write_check_image(d: Path, cal):
    """Every JSON hull pose drawn on its screenshot: the marks must sit on the rendered hulls."""
    import matplotlib.pyplot as plt
    pngs = sorted(p for p in d.glob("d*.png") if p.with_suffix(".json").exists() and "_preview" not in p.name)
    if not pngs:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 9)); axes = axes.ravel()
    for ax, png in zip(axes, pngs):
        cell = json.loads(png.with_suffix(".json").read_text()); ax.imshow(plt.imread(png))
        P = cal.to_pix([[g["x"], g["y"]] for g in cell["ghosts"]])
        ax.plot(P[:, 0], P[:, 1], "+", color="lime", ms=10, mew=1.5)
        for g, p in zip(cell["ghosts"], P):
            ax.annotate(f"{g['t_since_fault']:.0f}", p, xytext=(4, -4), textcoords="offset points", color="lime", fontsize=7)
        ax.set_title(cell["cell"]); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(d / "calibration_check.png", dpi=110); plt.close(fig)
    print(f"wrote {d / 'calibration_check.png'}: green crosses should sit on the hulls")


class Cal:
    def __init__(self, d: dict):
        self.A = np.asarray(d["A"]); self.b = np.asarray(d["b"]); self.W, self.H = d["image_size"]
        self.Ainv = np.linalg.inv(self.A)

    def to_pix(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float)); return xy @ self.A.T + self.b

    def to_world(self, px):
        px = np.atleast_2d(np.asarray(px, float)); return (px - self.b) @ self.Ainv.T


# ------------------------------------------------------------------ drawing
def crop_to_extent(img, cal: Cal, extent):
    """Crop the screenshot to a world extent; returns (crop, imshow_extent)."""
    x0, x1, y0, y1 = extent
    p = cal.to_pix([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    c0, r0 = np.floor(p.min(axis=0)).astype(int); c1, r1 = np.ceil(p.max(axis=0)).astype(int)
    c0, r0 = max(c0, 0), max(r0, 0); c1, r1 = min(c1, cal.W), min(r1, cal.H)
    crop = img[r0:r1, c0:c1]
    # world coordinates of the crop corners (axis-aligned assumption)
    tl = cal.to_world([[c0, r0]])[0]; br = cal.to_world([[c1, r1]])[0]
    return crop, (tl[0], br[0], br[1], tl[1])


def draw_harbor(ax, zone_alpha, labels, hull_radius):
    from matplotlib.patches import Polygon as MPoly
    for z in ZONES:
        v = np.asarray(z.vertices)
        ax.add_patch(MPoly(v, closed=True, facecolor="#5ee06a", edgecolor="#137a20", alpha=zone_alpha, lw=0.9, zorder=3))
        if labels:
            ax.text(*v.mean(axis=0), z.name, ha="center", va="center", fontsize=6, color="#0c4d14",
                    fontweight="bold", zorder=6)
    for i, d in enumerate(DOCK_VERTICES):
        v = np.asarray(d)
        ax.add_patch(MPoly(v, closed=True, facecolor="none", edgecolor="#d81e1e", hatch="////", lw=0.9, zorder=3))
        if labels:
            ax.text(*v.mean(axis=0), f"Dock {i + 1}", ha="center", va="center", fontsize=6, color="#8a0b0b",
                    fontweight="bold", zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        if hull_radius > 0:
            try:
                from shapely.geometry import Polygon as SPoly
                ring = np.asarray(SPoly(v).buffer(hull_radius, join_style=2).exterior.coords)
                ax.plot(ring[:, 0], ring[:, 1], color="#d81e1e", lw=0.5, ls=(0, (2, 2)), zorder=3)
            except ImportError:
                pass
    s = np.asarray(SHORE_POLYLINE)
    ax.plot(s[:, 0], s[:, 1], color="#5a3d1e", lw=0.8, zorder=3)


def terminal_marker(outcome: str):
    o = outcome.lower()
    if o.startswith("arrived"):
        return "o", "arrival in a berthing zone"
    if "contact" in o or "collision" in o:
        return "X", "pier collision"
    return "^", "stopped short"


def draw_events(ax, cell: dict, mark_ms):
    handles = {}
    g = cell["ghosts"][-1]
    m, lab = terminal_marker(cell["outcome"])
    ax.plot(g["x"], g["y"], m, ms=mark_ms, mfc="white", mec=C_TRACK, mew=1.2, zorder=7)
    handles[lab] = (m, C_TRACK)
    for name, b in cell.get("baselines", {}).items():
        tr = np.asarray(b["track"])
        if len(tr) == 0:
            continue
        m, lab = terminal_marker(b["outcome"])
        ax.plot(tr[-1, 0], tr[-1, 1], m, ms=mark_ms, mfc="white", mec=C_INTERNAL, mew=1.2, zorder=7)
        handles[lab] = (m, C_INTERNAL)
    return handles


def draw_timestamps(ax, cell: dict, every: int, fontsize):
    for i, g in enumerate(cell["ghosts"]):
        if i % every and g.get("tag") != "terminal":
            continue
        ax.annotate(f"{g['t_since_fault']:.0f} s", (g["x"], g["y"]), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=fontsize, color="white", zorder=8,
                    path_effects=_outline())


def _outline():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.6, foreground="black")]


def scale_bar(ax, extent, length, fontsize):
    x0, x1, y0, y1 = extent
    xs, ys = x1 - 0.06 * (x1 - x0) - length, y0 + 0.06 * (y1 - y0)
    ax.plot([xs, xs + length], [ys, ys], color="white", lw=2.2, solid_capstyle="butt", zorder=8, path_effects=_outline())
    ax.text(xs + length / 2, ys + 0.015 * (y1 - y0), f"{length:.0f} m", ha="center", va="bottom",
            fontsize=fontsize, color="white", zorder=8, path_effects=_outline())


def build_legend(fig, marker_handles, fontsize, ncol):
    from matplotlib.lines import Line2D
    items = [Line2D([], [], color=C_TRACK, lw=2.2, label="executed trajectory of the manifold planner"),
             Line2D([], [], color=C_INTERNAL, lw=1.4, label="executed trajectory of the reachability planner"),
             Line2D([], [], color=C_REF, lw=2.2, label="initial feasible plan"),
             Line2D([], [], color=C_REPLAN, lw=2.2, label="final replanned plan")]
    for lab in ("arrival in a berthing zone", "pier collision", "stopped short"):
        if lab in marker_handles:
            m, _ = marker_handles[lab]
            items.append(Line2D([], [], marker=m, ls="none", mfc="white", mec="black", mew=1.2, ms=6, label=lab))
    fig.legend(handles=items, loc="lower center", ncol=ncol, fontsize=fontsize, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.2, handletextpad=0.6)


# ------------------------------------------------------------------ caption
def write_caption(cells: dict, path: Path, spacing_note: str):
    lines = ["Closed-loop harbour return in Gazebo/VRX, sea state 3, port thruster degraded, one representative trial per cell. "
             "Top row: wind and waves aligned with the approach; bottom row: quartering seas. Columns: 50 %, 80 % and 95 % degradation. "
             "WAM-V models mark the ground-truth pose of the manifold arm at the fault, at regular intervals "
             f"({spacing_note}) and at the terminal event; the blue line is its track. Red is the certified decoded plan the sidecar "
             "published after the fault and orange the last plan it replanned to; the grey line is the ground truth of the "
             "reachability-based planner of [5] under the same seed and sea. Open circle: entry into a berthing zone; cross: pier "
             "collision; triangle: closest approach when the vessel stopped short. Green: berthing zones; hatched: piers; "
             "polygons are those used by the scorer. Panels share one scale."]
    per = []
    for (sev, d) in CELLS:
        c = cells.get(f"{sev}_{d}")
        if c:
            per.append(f"({'abcdef'[CELLS.index((sev, d))]}) {SEV_LABEL[sev]}, {DIR_LABEL[d]}: {c['trial_id']}, "
                       f"{c['outcome']} at {c['t_end_since_fault']:.0f} s"
                       + (f"; internal: {c['baselines']['internal']['outcome']}" if c.get("baselines", {}).get("internal") else ""))
    lines.append("\nPer-panel record (for the text, not the caption):\n" + "\n".join(per))
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=None, help="directory with <cell>.png, <cell>.json, calibration.json")
    p.add_argument("--calibrate-ghosts", nargs="+", default=None, metavar="PNG",
                   help="click the ghost hulls on these screenshots (world poses from the JSONs)")
    p.add_argument("--calibrate", default=None, metavar="PNG", help="interactive pier-corner picking (less accurate)")
    p.add_argument("--calibrate-from", default=None, metavar="TXT", help="'x y px py' lines instead of clicking")
    p.add_argument("--calibration", default=None, help="calibration.json (default <dir>/calibration.json)")
    p.add_argument("--extent", nargs=4, type=float, default=[-606, -446, 176, 266],
                   metavar=("X0", "X1", "Y0", "Y1"), help="world window shown in every panel (m)")
    p.add_argument("--hull-radius", type=float, default=0.0,
                   help="draw the pier polygons inflated by this radius as a dashed line (0 = off)")
    p.add_argument("--zone-alpha", type=float, default=0.28)
    p.add_argument("--timestamps", nargs="*", default=["d95"], help="severities that get time labels at the hulls")
    p.add_argument("--timestamp-every", type=int, default=2, help="label every n-th hull")
    p.add_argument("--scale-bar", type=float, default=20.0)
    p.add_argument("--width", type=float, default=7.16, help="figure width in inches (IEEE double column 7.16)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fontsize", type=float, default=7.0)
    p.add_argument("--label-pos", choices=["below", "inside"], default="below", help="panel tag placement")
    p.add_argument("--legend-cols", type=int, default=4)
    p.add_argument("--out", default="closed_loop")
    a = p.parse_args()

    d = Path(a.dir).expanduser() if a.dir else None
    if a.calibrate_ghosts:
        pngs = [Path(x).expanduser() for x in a.calibrate_ghosts]
        calibrate_ghosts(pngs, (d or pngs[0].parent) / "calibration.json")
        return
    if a.calibrate or a.calibrate_from:
        png = Path(a.calibrate or a.calibrate_from).expanduser()
        if a.calibrate_from:
            pngs = sorted((png.parent if d is None else d).glob("d*_beneficial.png"))
            if not pngs:
                sys.exit("give --dir so a screenshot can be found for the image size")
            calibrate_from_file(png, pngs[0], (d or png.parent) / "calibration.json")
        else:
            calibrate_interactive(png, (d or png.parent) / "calibration.json")
        return
    if d is None:
        sys.exit("--dir is required")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": a.fontsize, "pdf.fonttype": 42})

    cal = Cal(json.loads(Path(a.calibration or d / "calibration.json").read_text()))
    x0, x1, y0, y1 = a.extent
    aspect = (y1 - y0) / (x1 - x0)
    panel_w = (a.width - 2 * 0.06) / 3
    tag_h = 0.18 if a.label_pos == "below" else 0.0
    legend_h = 0.55
    fig_h = 2 * (panel_w * aspect + tag_h) + legend_h + 0.04
    fig, axes = plt.subplots(2, 3, figsize=(a.width, fig_h))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=(legend_h + tag_h) / fig_h,
                        wspace=0.025, hspace=(tag_h + 0.03) / (panel_w * aspect))

    marker_handles, cells, spacing = {}, {}, set()
    for k, (sev, direction) in enumerate(CELLS):
        ax = axes[k // 3, k % 3]; name = f"{sev}_{direction}"
        png, js = d / f"{name}.png", d / f"{name}.json"
        if not (png.exists() and js.exists()):
            ax.text(0.5, 0.5, f"missing {name}", ha="center", va="center", transform=ax.transAxes); ax.set_axis_off(); continue
        cell = json.loads(js.read_text()); cells[name] = cell
        gh = cell["ghosts"]
        if len(gh) > 2:
            spacing.add(round(gh[2]["t_since_fault"] - gh[1]["t_since_fault"]))
        img = plt.imread(png)
        crop, ext = crop_to_extent(img, cal, a.extent)
        ax.imshow(crop, extent=ext, zorder=1, interpolation="lanczos")
        draw_harbor(ax, a.zone_alpha, labels=(k == 0), hull_radius=a.hull_radius)
        marker_handles.update(draw_events(ax, cell, mark_ms=6))
        # if sev in a.timestamps:
        #     draw_timestamps(ax, cell, a.timestamp_every, a.fontsize - 1)
        if k == 0:
            scale_bar(ax, a.extent, a.scale_bar, a.fontsize)
        tag = f"({'abcdef'[k]}) {SEV_LABEL[sev]} degradation, {DIR_LABEL[direction]}"
        if a.label_pos == "below":
            ax.text(0.5, -0.015, tag, transform=ax.transAxes, ha="center", va="top", fontsize=a.fontsize, zorder=9)
        else:
            ax.text(0.015, 0.975, tag, transform=ax.transAxes, ha="left", va="top", fontsize=a.fontsize,
                    fontweight="bold", zorder=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_axis_off()

    build_legend(fig, marker_handles, a.fontsize, ncol=a.legend_cols)
    for ext_ in ("pdf", "png"):
        fig.savefig(d / f"{a.out}.{ext_}", dpi=a.dpi)
    note = " and ".join(f"every {s} s" for s in sorted(spacing)) or "regular intervals"
    write_caption(cells, d / "caption.txt", note)
    print(f"wrote {d / a.out}.pdf, .png and caption.txt")


if __name__ == "__main__":
    main()
