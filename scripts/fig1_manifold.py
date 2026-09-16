#!/usr/bin/env python3
"""Fig. 1: uninformed proposal versus learned recovery manifold at 95 %
port-thruster degradation.

Two stacked single-column panels sharing the start pose, the harbour, the
health and ONE force draw, so that the generator is the only thing that
changes between them: (a) K samples from the naive proposal of Section III-E
(what rejection sampling draws from), (b) K candidates decoded from the
conditioned model. Both are screened with the same feasibility screen as
everything else in the paper and drawn as a fan: feasible candidates coloured
by target zone (Okabe-Ito, as in the motivation geometry), rejected candidates
in light grey. --panels model:0.5 model:0.05 gives the old severity contrast.

    python scripts/fig1_manifold.py --out runs/fig1_manifold          # real decode
    python scripts/fig1_manifold.py --out /tmp/fig1 --placeholder     # layout only

Writes <out>.pdf, <out>.png and prints the per-panel statistics (feasible
fraction, feasible count per zone, class count, failure breakdown) so the
figure can be checked against Table I and Section V-C before it goes in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MPoly, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from manifold_recovery.scenario import (ZONES, DOCK_VERTICES, LAND_VERTICES, SPIKE_START_POSE,  # noqa: E402
                                        zone_of)

# ---- frame and style (matches fig1_motivation.py) -------------------------
XMIN, XMAX, YMIN, YMAX = -610.0, -446.0, 178.0, 266.0
FIG_W = 3.45
C_WATER, C_LAND, C_LAND_EDGE = "#EAF0F4", "#D9D2C8", "#A89F91"
C_PIER, C_PIER_EDGE = "#F4E1DD", "#B03A2E"          # same hatch convention as the closed-loop figure
C_ZONE, C_ZONE_EDGE = "#D5EBD9", "#3C8A45"
C_VESSEL, C_REJECT, C_BASE, C_FAULT = "#3A3A3A", "#C9CED3", "#6E6E6E", "#D62728"
ZONE_NAME = {0: "Zone 1", 1: "Zone 2", 2: "Zone 3"}
RAMP, RAMP_LO, RAMP_HI = "viridis", 0.08, 0.90        # feasible candidates: colour = lateral offset
INTERNAL_TARGET = np.array([-570.0, 223.0])     # corner of Zone 2 targeted by the planner of [5]


def clip(poly):
    from shapely.geometry import Polygon as SP, box
    g = SP(np.asarray(poly, float)).buffer(0).intersection(box(XMIN, YMIN, XMAX, YMAX))
    return [np.asarray(gg.exterior.coords)[:-1] for gg in getattr(g, "geoms", [g]) if gg.geom_type == "Polygon"]


def draw_harbour(ax, labels: bool, fs: float):
    ax.add_patch(Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN, facecolor=C_WATER, edgecolor="none", zorder=0))
    for v in LAND_VERTICES:
        for ring in clip(v):
            ax.add_patch(MPoly(ring, closed=True, facecolor=C_LAND, edgecolor=C_LAND_EDGE, lw=0.5, zorder=1))
    for i, v in enumerate(DOCK_VERTICES):
        ax.add_patch(MPoly(np.asarray(v), closed=True, facecolor=C_PIER, edgecolor=C_PIER_EDGE, lw=0.6,
                           hatch="///", zorder=2))
        if labels:
            ax.text(*np.mean(v, axis=0), f"Dock {i + 1}", ha="center", va="center", fontsize=fs, color="#7A2A20",
                    fontweight="bold", zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for z in ZONES:
        ax.add_patch(MPoly(np.asarray(z.vertices), closed=True, facecolor=C_ZONE, edgecolor=C_ZONE_EDGE, lw=0.6, zorder=2))
        if labels:
            ax.text(*np.mean(z.vertices, axis=0), ZONE_NAME[z.id], ha="center", va="center", fontsize=fs,
                    color="#1E5A28", fontweight="bold", zorder=6)


def vessel_glyph(ax, centre, heading, scale):
    """Twin-pontoon WAM-V from above; body y to starboard (Fossen). Returns the port thruster anchor."""
    hull_l, pont_w, sep = 4.9, 1.0, 2.05
    c, s = np.cos(heading), np.sin(heading); R = np.array([[c, -s], [s, c]])
    place = lambda pts: (np.asarray(pts, float) * scale) @ R.T + centre
    for side in (+1, -1):
        y0 = side * sep / 2
        body = [[hull_l / 2, y0 - pont_w / 2], [hull_l / 2 - 0.9, y0 + pont_w / 2],
                [-hull_l / 2, y0 + pont_w / 2], [-hull_l / 2, y0 - pont_w / 2]]
        ax.add_patch(MPoly(place(body), closed=True, facecolor=C_VESSEL, edgecolor="none", zorder=7))
    ax.add_patch(MPoly(place([[0.9, -sep / 2], [0.9, sep / 2], [-1.4, sep / 2], [-1.4, -sep / 2]]),
                       closed=True, facecolor=C_VESSEL, alpha=0.55, edgecolor="none", zorder=7))
    return place([[-hull_l / 2 + 0.2, -sep / 2]])[0]


def straight_route(x0, field, hull_radius, n=300):
    """Chord from the start to the Zone 2 corner; returns the path and its first contact point (or None)."""
    p = np.linspace(x0[:2], INTERNAL_TARGET, n)
    d = field.signed_distance(p[None])[0]
    hit = np.flatnonzero(d < hull_radius)
    return p, (p[hit[0]] if len(hit) else None)


# ---- data -----------------------------------------------------------------
def parse_panels(specs):
    out = []
    for sp in specs:
        kind, h1 = sp.split(":"); out.append((kind, float(h1)))
    return out


def decode_and_screen(panel_specs, K, seed, sea_state, direction, config, ckpt):
    from manifold_recovery.config import load
    from manifold_recovery.traj.rtp import RTP
    from manifold_recovery.score.obstacles import DockField
    from manifold_recovery.certify.surrogate import certify_batch
    from manifold_recovery.data.env_forces import SyntheticSampler
    from manifold_recovery.model.train import load as load_ckpt
    from manifold_recovery.model.decode import Decoder
    from manifold_recovery.data.proposal import ProposalSampler
    cfg = load(config); model, meta = load_ckpt(ckpt)
    if meta.get("config_hash") != cfg.hash():
        print("WARNING: checkpoint config hash differs from the current config")
    dec, rtp, field = Decoder(model, meta, cfg), RTP(cfg.trajectory), DockField()
    T_h, dt = rtp.horizon()
    rng = np.random.default_rng(seed)
    w, sig = SyntheticSampler().sample(sea_state, direction, T_h, dt, rng)
    w = w[: cfg.trajectory.N]
    x0 = np.asarray(SPIKE_START_POSE, float)
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, mix=cfg.data.prop_mix)
    panels = []
    for kind, h1 in panel_specs:
        if kind == "proposal":
            om, g = prop.sample_with_mode(np.broadcast_to(x0, (K, 3)), rng)
        else:
            om, g = dec.decode_zones(h1, x0, K, rng)
        cert = certify_batch(om, x0, h1, 1.0, np.broadcast_to(w, (len(om),) + w.shape), sig, rtp, field, cfg)
        pos = rtp.kinematics(om, x0).pos
        panels.append(dict(kind=kind, h1=h1, pos=np.asarray(pos), zone=np.asarray(g), mask=np.asarray(cert.mask),
                           breakdown=cert.failure_breakdown()))
    return x0, panels, field, float(cfg.exact.hull_radius)


def placeholder(panel_specs, K, seed):
    """Synthetic fans for layout work only; never for the paper."""
    from manifold_recovery.score.obstacles import DockField
    rng = np.random.default_rng(seed); x0 = np.asarray(SPIKE_START_POSE, float)
    t = np.linspace(0, 1, 80)[:, None]
    panels = []
    for kind, h1 in panel_specs:
        pos, zone, mask = [], [], []
        for k in range(K):
            z = k % 3; c = np.mean(ZONES[z].vertices, axis=0) + rng.normal(0, 2.5, 2)
            bulge = rng.normal(0, 9.0 if kind == "proposal" else 5.0) * np.sin(np.pi * t)
            p = (1 - t) * x0[:2] + t * c + np.column_stack([np.zeros_like(bulge), bulge])
            pos.append(p); zone.append(z)
            mask.append(rng.random() < (0.17 if kind == "proposal" else 0.55))
        panels.append(dict(kind=kind, h1=h1, pos=np.array(pos), zone=np.array(zone), mask=np.array(mask), breakdown={}))
    return x0, panels, DockField(), 1.5


def lateral_offset(xy):
    """Signed distance of the mid-path point from the start-to-end chord."""
    a, b, m = xy[0], xy[-1], xy[len(xy) // 2]
    d = b - a; n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)
    return float((m - a) @ n)


# ---- figure ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "runs/fig1_manifold"))
    p.add_argument("--config", default=str(ROOT / "configs/spike.yaml"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/ckpt.pt"))
    p.add_argument("--panels", nargs="+", default=["proposal:0.05", "model:0.05"],
                   help="panel specs kind:h1, kind in {proposal, model}")
    p.add_argument("--K", type=int, default=60)
    p.add_argument("--seed", type=int, default=7, help="seed for the force draw and the latent jitter")
    p.add_argument("--sea-state", type=int, default=3)
    p.add_argument("--direction", default="beneficial")
    p.add_argument("--baseline-panel", choices=["none", "last", "all"], default="none",
                   help="where to draw the straight route to the Zone 2 corner")
    p.add_argument("--glyph-scale", type=float, default=2.0, help="vessel glyph oversize factor (state in caption)")
    p.add_argument("--lw", type=float, default=0.8)
    p.add_argument("--fontsize", type=float, default=7.0)
    p.add_argument("--placeholder", action="store_true")
    p.add_argument("--ramp", default=None, help="sequential colormap for feasible candidates (default cividis)")
    a = p.parse_args()

    global RAMP
    if a.ramp:
        RAMP = a.ramp
    specs = parse_panels(a.panels)
    if a.placeholder:
        x0, panels, field, hull_r = placeholder(specs, a.K, a.seed)
    else:
        x0, panels, field, hull_r = decode_and_screen(specs, a.K, a.seed, a.sea_state, a.direction, a.config, a.ckpt)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": a.fontsize, "pdf.fonttype": 42, "hatch.linewidth": 0.5})
    aspect = (YMAX - YMIN) / (XMAX - XMIN)
    n = len(panels)
    legend_h = 0.52
    fig_h = FIG_W * aspect * n + 0.02 * (n - 1) + legend_h
    fig, axes = plt.subplots(n, 1, figsize=(FIG_W, fig_h))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=legend_h / fig_h, hspace=0.02 / (FIG_W * aspect))

    import matplotlib.colors as mcolors
    all_off = np.concatenate([[lateral_offset(xy) for xy in P["pos"][P["mask"]]] for P in panels] or [[0.0]])
    vmax = float(np.max(np.abs(all_off))) if len(all_off) else 1.0
    vmax = max(vmax, 1.0)
    norm = mcolors.Normalize(-vmax, vmax)
    base = plt.get_cmap(RAMP)
    cmap = mcolors.LinearSegmentedColormap.from_list("ramp", base(np.linspace(RAMP_LO, RAMP_HI, 256)))

    for k, (ax, P) in enumerate(zip(axes, panels)):
        draw_harbour(ax, labels=(k == 0), fs=a.fontsize)
        pos, zone, mask = P["pos"], P["zone"], P["mask"]
        for xy in pos[~mask]:                                             # rejected first, underneath
            ax.plot(xy[:, 0], xy[:, 1], color=C_REJECT, lw=a.lw * 0.8, alpha=0.7, zorder=3, solid_capstyle="round")
        fpos = pos[mask]
        off = np.array([lateral_offset(xy) for xy in fpos]) if len(fpos) else np.zeros(0)
        for xy, o in zip(fpos, off):
            ax.plot(xy[:, 0], xy[:, 1], color=cmap(norm(o)), lw=a.lw, alpha=0.9, zorder=4, solid_capstyle="round")
        if a.baseline_panel == "all" or (a.baseline_panel == "last" and k == n - 1):
            path, hit = straight_route(x0, field, hull_r)
            end = hit if hit is not None else path[-1]
            seg = path[: np.argmin(np.hypot(*(path - end).T)) + 1]
            ax.plot(seg[:, 0], seg[:, 1], color=C_BASE, lw=1.1, ls=(0, (4, 2.5)), zorder=5)
            if hit is not None:
                ax.plot(*hit, "X", ms=6, color=C_FAULT, mec="white", mew=0.5, zorder=8)
        port = vessel_glyph(ax, x0[:2], x0[2], a.glyph_scale)
        ax.plot(*port, "X", ms=6, color=C_FAULT, mec="white", mew=0.5, zorder=9)
        sev = int(round(100 * (1 - P["h1"])))
        who = "proposal distribution" if P["kind"] == "proposal" else "learned manifold"
        same_h = len({q["h1"] for q in panels}) == 1
        tag = f"({'abcd'[k]}) {who}" + ("" if same_h else f", {sev} % degraded")
        ax.text(XMAX - 2, YMAX - 2, tag, ha="right", va="top",
                fontsize=a.fontsize, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        ax.text(XMIN + 48, YMAX - 2, f"feasible {mask.sum()}/{len(mask)}", ha="left", va="top",
                fontsize=a.fontsize - 0.5, zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect("equal"); ax.set_axis_off()

        # statistics for the caption and for the check against Table I / Sec. V-C
        nz = {f"Z{z + 1}": (int((mask & (zone == z)).sum()), int((zone == z).sum())) for z in range(3)}
        classes = len(set(int(zone_of(xy[-1])) for xy in pos[mask]))
        print(f"{P['kind']} h1={P['h1']:.2f}: feasible {mask.mean():.2f} ({mask.sum()}/{len(mask)}), per zone {nz}, "
              f"classes among feasible {classes}, breakdown {P['breakdown']}")

    # legend below both panels, scale bar in the last panel
    items = [Line2D([], [], color=C_REJECT, lw=1.3, label="rejected by the feasibility screen"),
             Line2D([], [], marker="X", ls="none", color=C_FAULT, mec="white", mew=0.5, ms=6, label="degraded thruster")]
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.10, 0.075, 0.36, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("lateral offset of feasible candidates (m)", fontsize=a.fontsize - 1.0, labelpad=2)
    cb.ax.tick_params(labelsize=a.fontsize - 1.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
    if a.baseline_panel != "none":
        items.append(Line2D([], [], color=C_BASE, lw=1.1, ls=(0, (4, 2.5)), marker="X", mfc=C_FAULT, mec="white",
                            markevery=[1], ms=5, label="straight route to Z2"))
    fig.legend(handles=items, loc="lower right", ncol=1, fontsize=a.fontsize - 1.0, frameon=False,
               bbox_to_anchor=(0.995, 0.005), handlelength=2.0, handletextpad=0.5, labelspacing=0.3)
    ax = axes[-1]; L = 20.0; xs, ys = XMAX - 8 - L, YMIN + 5
    ax.plot([xs, xs + L], [ys, ys], color="black", lw=1.5, zorder=9)
    ax.text(xs + L / 2, ys + 1.5, f"{L:.0f} m", ha="center", va="bottom", fontsize=a.fontsize, zorder=9)

    out = Path(a.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=400)
    print(f"wrote {out}.pdf / .png  (glyph oversize x{a.glyph_scale}; K={a.K}; seed {a.seed}; "
          f"sea state {a.sea_state} {a.direction})")


if __name__ == "__main__":
    main()
