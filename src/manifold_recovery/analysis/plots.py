"""Figures for the spike report and the paper drafts."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..scenario import ZONES, DOCK_VERTICES


def _draw_harbor(ax):
    for v in DOCK_VERTICES:
        V = np.asarray(v + (v[0],))
        ax.fill(V[:, 0], V[:, 1], color="salmon", alpha=0.6, ec="firebrick",
                hatch="//", label=None)
    for z in ZONES:
        V = np.asarray(z.vertices + (z.vertices[0],))
        ax.fill(V[:, 0], V[:, 1], color="palegreen", alpha=0.5, ec="green")
        ax.annotate(z.name, z.center, ha="center", fontsize=8)
    ax.axvline(-570, color="gray", ls="--", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_trajectories(trajs, colors, title, out, x0=None, lw=1.0, cbar_label=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_harbor(ax)
    sm = None
    if np.ndim(colors) and len(colors) == len(trajs):
        cmap = plt.cm.viridis
        norm = plt.Normalize(np.min(colors), np.max(colors))
        for xi, c in zip(trajs, colors):
            ax.plot(xi[:, 0], xi[:, 1], color=cmap(norm(c)), lw=lw, alpha=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    else:
        for xi in trajs:
            ax.plot(xi[:, 0], xi[:, 1], color=colors, lw=lw, alpha=0.8)
    if x0 is not None:
        ax.plot(*x0[:2], "k*", ms=12)
    if sm is not None:
        fig.colorbar(sm, ax=ax, label=cbar_label or "")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_V_heatmap(df, value, out, title=None):
    piv = df.pivot_table(index="sea_state", columns="degradation", values=value)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=max(1.0, np.nanmax(piv.values)),
                   aspect="auto", origin="upper")
    ax.set_xticks(range(len(piv.columns)),
                  [f"{int(c*100)}%" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("Thruster degradation")
    ax.set_ylabel("Sea state")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=value)
    ax.set_title(title or value)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_history(hist, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    axs[0].plot(hist["recon"]); axs[0].set_title("weighted recon"); axs[0].set_yscale("log")
    axs[1].plot(hist["kl"], label="KL"); axs[1].plot(hist["Cz"], "--", label="Cz")
    axs[1].legend(); axs[1].set_title("KL vs capacity")
    for a in axs:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
