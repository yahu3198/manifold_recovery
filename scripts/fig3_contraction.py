"""Fig. 3 of the ICRA 2027 paper: manifold contraction at the nominal start.

Three panels (V, V_proposal, lift = V / V_proposal) over thruster degradation
x sea state, drawn from the aggregated table written by 04_contraction_map.py
(runs/maps.csv, canonical start). Nothing is recomputed, so the figure always
matches the numbers quoted in Section IV-B.

Changes relative to plot_V_heatmap in 04_contraction_map.py:
  * one figure with three panels sized for an IEEE two-column figure*,
  * the 100 % column is dropped (--max-degradation, default 0.95),
  * colour scales are shared (see --scale), no per-panel titles or long labels,
  * serif fonts at 8 pt, cell annotations with contrast-aware text colour,
  * vector PDF plus a 600 dpi PNG.

--scale split  (default) V and V_proposal share one [0, 1] scale and colourbar;
               lift has its own scale starting at 1 (no gain) with a second
               colourbar. Recommended: V and lift are different quantities.
--scale shared all three panels on one [0, max] scale with a single colourbar.

Usage:
  python scripts/fig3_contraction.py
  python scripts/fig3_contraction.py --csv runs/maps.csv --out figures/fig3_contraction
  python scripts/fig3_contraction.py --scale shared
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--csv", default=str(ROOT / "runs/maps.csv"),
               help="aggregated canonical-start table from 04_contraction_map.py")
p.add_argument("--out", default=str(ROOT / "figures/fig3_contraction"),
               help="output path without extension (.pdf and .png are written)")
p.add_argument("--scale", choices=("split", "shared"), default="split")
p.add_argument("--max-degradation", type=float, default=0.95,
               help="drop severity columns above this value (removes 100 %%)")
p.add_argument("--width", type=float, default=7.16, help="inches (IEEE text width)")
p.add_argument("--height", type=float, default=1.55, help="inches")
p.add_argument("--cmap-v", default="RdYlGn")
p.add_argument("--cmap-lift", default="Blues")
p.add_argument("--no-annot", action="store_true")
a = p.parse_args()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "pdf.fonttype": 42, "ps.fonttype": 42,        # embed TrueType (IEEE PDF eXpress)
})

# ---------------------------------------------------------------- data
df = pd.read_csv(a.csv)
df["degradation"] = df["degradation"].round(4)
df = df[df["degradation"] <= a.max_degradation + 1e-9]
if "direction" in df and df.groupby(["degradation", "sea_state"]).size().max() > 1:
    print("WARNING: several directions per cell, values are averaged over them")


def grid(col):
    piv = df.pivot_table(index="sea_state", columns="degradation", values=col, aggfunc="mean")
    return piv.sort_index().sort_index(axis=1)


panels = [("V", r"$V$"), ("V_proposal", r"$V_{\mathrm{prop}}$"),
          ("lift", r"$L = V / V_{\mathrm{prop}}$")]
grids = {c: grid(c) for c, _ in panels}
sev, ss = grids["V"].columns.to_numpy(), grids["V"].index.to_numpy()
for c, g in grids.items():
    assert np.array_equal(g.columns, sev) and np.array_equal(g.index, ss), c

# ---------------------------------------------------------------- scales
lift_max = float(np.nanmax(grids["lift"].values))
if a.scale == "split":
    cmap_v = plt.get_cmap(a.cmap_v)
    norm_v = mcolors.Normalize(0.0, 1.0)
    cmap_l = plt.get_cmap(a.cmap_lift)
    norm_l = mcolors.Normalize(1.0, np.ceil(lift_max * 2) / 2)   # clip <1 to "no gain"
    style = {"V": (cmap_v, norm_v), "V_proposal": (cmap_v, norm_v), "lift": (cmap_l, norm_l)}
else:
    cmap_s = plt.get_cmap("viridis")
    norm_s = mcolors.Normalize(0.0, np.ceil(lift_max * 2) / 2)
    style = {c: (cmap_s, norm_s) for c, _ in panels}

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(a.width, a.height))
if a.scale == "split":
    # [V][Vp][cbar]  gap  [L][cbar]
    wr = [1, 1, 0.05, 0.30, 1, 0.05]
    gs = GridSpec(1, 6, figure=fig, width_ratios=wr, wspace=0.12,
                  left=0.05, right=0.93, bottom=0.25, top=0.88)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
    caxes = {"V": fig.add_subplot(gs[0, 2]), "lift": fig.add_subplot(gs[0, 5])}
else:
    wr = [1, 1, 1, 0.05]
    gs = GridSpec(1, 4, figure=fig, width_ratios=wr, wspace=0.12,
                  left=0.05, right=0.93, bottom=0.25, top=0.88)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    caxes = {"V": fig.add_subplot(gs[0, 3])}

ims = {}
for k, (ax, (col, label)) in enumerate(zip(axes, panels)):
    Z = grids[col].values
    cmap, norm = style[col]
    ims[col] = ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto", origin="upper",
                         interpolation="nearest")
    ax.set_xticks(range(len(sev)), [f"{round(s * 100)}%" for s in sev])
    ax.set_yticks(range(len(ss)), [str(int(s)) for s in ss])
    ax.tick_params(length=0)
    ax.set_title(label, pad=2.5)
    ax.set_xlabel("Thruster degradation", labelpad=1.5)
    if k == 0:
        ax.set_ylabel("Sea state", labelpad=1.5)
    elif not (a.scale == "split" and col == "lift"):
        ax.set_yticklabels([])      # lift panel keeps its tick labels after the gap
    for s in ax.spines.values():
        s.set_visible(False)
    if not a.no_annot:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = cmap(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if lum < 0.5 else "black")

cb = fig.colorbar(ims["V"], cax=caxes["V"])
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=6.5, length=1.5, width=0.4)
if a.scale == "split":
    cb.set_label("Feasible fraction", labelpad=2)
    cbl = fig.colorbar(ims["lift"], cax=caxes["lift"])
    cbl.outline.set_linewidth(0.4)
    cbl.ax.tick_params(labelsize=6.5, length=1.5, width=0.4)
    cbl.set_label("Lift", labelpad=2)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=600)
plt.close(fig)

print(f"wrote {out.with_suffix('.pdf')} and .png  (scale={a.scale})")
for c, _ in panels:
    print(f"\n# {c}")
    print(grids[c].rename(columns=lambda s: f"{round(s * 100)}%")
          .to_string(float_format=lambda v: f"{v:.2f}"))
