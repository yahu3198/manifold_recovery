#!/usr/bin/env python3
"""Ghost-pose extraction for the closed-loop result figure (ICRA 2027, Fig. 5).

For each cell (degradation x direction) of one arm, pick a representative trial
from trials.csv, read its bag, and write everything the Gazebo ghost scene needs:

  * ghosts : poses (x, y, psi) at the fault, at reference receipt, every
             --spacing seconds, and at the terminal instant (zone entry, first
             pier contact, or closest approach to a zone when the vessel never
             arrives)
  * track  : ground-truth position after the fault, resampled every --track-step m
  * refs   : every reference published on /wamv/manifold_ref (stage 1 and each
             replan), as (x, y) polylines
  * world  : the trial world file and its <world name>, for the driver

Selection rule (stated in the caption): at 50 % and 80 %, the arrival trial with
the median time to zone under the 400 s criterion; at 95 %, the trial with the
median closest approach to a zone. Override with --seed CELL=SEED.

    python scripts/06_ghost_poses.py --trials ~/usv_ws/experiments/scored/trials.csv \
        --bags ~/usv_ws/experiments/bags --out ~/usv_ws/experiments/figure_ghosts

Writes <out>/<cell>.json and <out>/<cell>_preview.png. Pure Python (rosbags);
no ROS needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manifold_recovery.scenario import ZONES, zone_of                 # noqa: E402
from manifold_recovery.score.obstacles import DockField               # noqa: E402
from manifold_recovery.io_bridge.bag_reader import read_bag           # noqa: E402

EXT_S = 400.0                       # secondary time criterion, as in score_trials.py
DIR_LABEL = {"beneficial": "aligned", "nominal": "quartering"}
DEFAULT_SPACING = {0.5: 25.0, 0.8: 30.0, 0.95: 40.0}


# ---------------------------------------------------------------- selection
def pick_trial(df: pd.DataFrame, arm: str, deg: float, direction: str) -> pd.Series:
    sub = df[(df.arm == arm) & (np.isclose(df.degradation, deg)) & (df.direction == direction)]
    if "error" in sub:
        sub = sub[sub["error"].isna()]
    if sub.empty:
        raise SystemExit(f"no scored trials for {arm} d{int(deg*100)} {direction}")
    if deg < 0.9:
        ok = sub[sub.success_ext.astype(bool)]
        if len(ok):
            return ok.iloc[(ok.time_to_zone_s - ok.time_to_zone_s.median()).abs().argsort().iloc[0]]
        print(f"  no arrival in {arm} d{int(deg*100)} {direction}; falling back to closest approach")
    col = "dist_to_zone_min_m"
    return sub.iloc[(sub[col] - sub[col].median()).abs().argsort().iloc[0]]


# ---------------------------------------------------------------- bag reading
def read_refs(bag: Path):
    """All /wamv/manifold_ref messages as (t_msg, t0, dt, rows[N, 8])."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    out = []
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = [c for c in reader.connections if c.topic == "/wamv/manifold_ref"]
        for conn, ts, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            data = np.asarray(m.data, float)
            if len(data) < 2 + 8:
                continue
            rows = data[2:].reshape(-1, 8)
            out.append((ts * 1e-9, float(data[0]), float(data[1]), rows))
    return out


def resample_by_arclength(xy: np.ndarray, step: float) -> np.ndarray:
    if len(xy) < 2:
        return xy
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    if s[-1] < step:
        return xy[[0, -1]]
    si = np.arange(0.0, s[-1], step)
    si = np.append(si, s[-1])
    return np.column_stack([np.interp(si, s, xy[:, 0]), np.interp(si, s, xy[:, 1])])


def pose_at(t: float, t_o, pos, psi_unwrapped) -> tuple[float, float, float]:
    x = float(np.interp(t, t_o, pos[:, 0]))
    y = float(np.interp(t, t_o, pos[:, 1]))
    p = float(np.interp(t, t_o, psi_unwrapped))
    return x, y, float((p + np.pi) % (2 * np.pi) - np.pi)


def world_name(world_sdf: Path) -> str:
    try:
        m = re.search(r"<world\s+name\s*=\s*\"([^\"]+)\"", world_sdf.read_text())
        return m.group(1) if m else "default"
    except OSError:
        return "default"


def terminal_index(pos: np.ndarray, field: DockField):
    """First zone entry, else first pier contact, else closest approach; same rules as score_trials.py."""
    z = zone_of(pos); hit = np.flatnonzero(z >= 0)
    sd = field.signed_distance(pos); col = np.flatnonzero(sd < 0)
    try:
        from shapely.geometry import Polygon, Point
        polys = [Polygon(zn.vertices) for zn in ZONES]
        dz = np.array([min(pg.distance(Point(x, y)) for pg in polys) for x, y in pos])
    except Exception:
        dz = np.full(len(pos), np.nan)
    if len(hit) and (not len(col) or hit[0] <= col[0]):
        return int(hit[0]), f"arrived zone {int(z[hit[0]]) + 1}"
    if len(col):
        return int(col[0]), "pier contact"
    return int(np.nanargmin(dz)), f"closest approach {float(np.nanmin(dz)):.1f} m"


def extract_track(bag: Path, track_step: float, field: DockField) -> dict:
    """Ground-truth track only (for baseline arms drawn as a line, no ghosts)."""
    d = read_bag(bag)
    fi = np.flatnonzero(d["h"].min(axis=1) < 99.9)
    if len(fi) == 0:
        raise RuntimeError("no fault in bag")
    t_f = float(d["t_h"][fi[0]])
    sel = (d["t_o"] >= t_f) & (d["t_o"] <= t_f + EXT_S)
    pos = d["pos"][sel]
    i_end, outcome = terminal_index(pos, field)
    return {"outcome": outcome, "t_end_since_fault": float(d["t_o"][sel][i_end] - t_f),
            "track": resample_by_arclength(pos[: i_end + 1], track_step).tolist()}


def extract(bag: Path, manifest: dict | None, spacing: float, track_step: float, field: DockField):
    d = read_bag(bag)
    if len(d["t_h"]) == 0 or len(d["t_o"]) == 0:
        raise RuntimeError("bag lacks health or odometry")
    fi = np.flatnonzero(d["h"].min(axis=1) < 99.9)
    if len(fi) == 0:
        raise RuntimeError("no fault in bag")
    t_f = float(d["t_h"][fi[0]])
    h = (d["h"][fi[0]] / 100.0).tolist()

    sel = (d["t_o"] >= t_f) & (d["t_o"] <= t_f + EXT_S)
    t_o, pos, psi = d["t_o"][sel], d["pos"][sel], np.unwrap(d["psi"][sel])

    i_end, outcome = terminal_index(pos, field)
    t_end = float(t_o[i_end])

    # published references
    refs = []
    for t_msg, t0, dt, rows in read_refs(bag):
        if t_msg < t_f - 1.0:
            continue
        refs.append({"t_since_fault": t_msg - t_f, "t0_since_fault": t0 - t_f, "dt": dt,
                     "xy": rows[:, :2].tolist(), "psi": rows[:, 2].tolist()})
    t_ref = t_f + refs[0]["t_since_fault"] if refs else None

    # ghost instants: fault, reference receipt, regular spacing, terminal
    times = [t_f]
    if t_ref is not None and t_ref - t_f > 2.0:
        times.append(t_ref)
    tk = t_f + spacing
    while tk < t_end - 0.5 * spacing:
        times.append(tk); tk += spacing
    times.append(t_end)
    times = sorted(set(round(t, 3) for t in times))
    tags = {round(t_f, 3): "fault", round(t_end, 3): "terminal"}
    if t_ref is not None:
        tags.setdefault(round(t_ref, 3), "reference")
    ghosts = []
    for t in times:
        x, y, p = pose_at(t, t_o, pos, psi)
        ghosts.append({"t_since_fault": t - t_f, "x": x, "y": y, "psi": p, "tag": tags.get(t, "")})

    track = resample_by_arclength(pos[: i_end + 1], track_step)
    world = Path(manifest["world"]) if manifest and "world" in manifest else None
    return {"t_fault": t_f, "H": h, "t_end_since_fault": t_end - t_f, "outcome": outcome,
            "ghosts": ghosts, "track": track.tolist(), "refs": refs,
            "world": str(world) if world else None,
            "world_name": world_name(world) if world else None}


# ---------------------------------------------------------------- preview
def preview(cell: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from manifold_recovery.analysis.plots import _draw_harbor
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_harbor(ax)
    for name, b in cell.get("baselines", {}).items():
        bt = np.asarray(b["track"])
        if len(bt):
            ax.plot(bt[:, 0], bt[:, 1], color="0.45", lw=0.9, ls="--", label=f"{name} planner ({b['outcome']})")
    refs = cell["refs"]
    shown = refs if len(refs) <= 2 else [refs[0], refs[-1]]
    for i, r in enumerate(shown):
        xy = np.asarray(r["xy"])
        ax.plot(xy[:, 0], xy[:, 1], color="red" if i == 0 else "darkorange", lw=0.9,
                label="initial certified plan" if i == 0 else "final replanned plan")
    tr = np.asarray(cell["track"])
    if len(tr):
        ax.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1.4, label="ground truth")
    for g in cell["ghosts"]:
        L = 4.9
        ax.plot([g["x"] - 0.5 * L * np.cos(g["psi"]), g["x"] + 0.5 * L * np.cos(g["psi"])],
                [g["y"] - 0.5 * L * np.sin(g["psi"]), g["y"] + 0.5 * L * np.sin(g["psi"])],
                color="k", lw=3, alpha=0.7)
        ax.annotate(f"{g['t_since_fault']:.0f}s", (g["x"], g["y"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    xs = np.array([g["x"] for g in cell["ghosts"]]); ys = np.array([g["y"] for g in cell["ghosts"]])
    ax.set_xlim(-612, max(xs.max(), -560) + 10); ax.set_ylim(min(ys.min(), 178) - 6, max(ys.max(), 262) + 6)
    ax.set_title(f"{cell['cell']}: {cell['trial_id']}  ({cell['outcome']})", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", required=True, help="trials.csv from score_trials.py")
    p.add_argument("--bags", required=True)
    p.add_argument("--manifests", default=None, help="manifests dir (default: <bags>/../manifests)")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="manifold")
    p.add_argument("--degrade", nargs="+", type=float, default=[0.5, 0.8, 0.95])
    p.add_argument("--directions", nargs="+", default=["beneficial", "nominal"])
    p.add_argument("--spacing", nargs="*", default=[],
                   help="per-severity ghost spacing in s, e.g. 0.5=25 0.8=30 0.95=40")
    p.add_argument("--track-step", type=float, default=2.0)
    p.add_argument("--seed", nargs="*", default=[], help="override: CELL=SEED, e.g. d80_nominal=0353")
    p.add_argument("--baselines", nargs="*", default=["internal"],
                   help="other arms whose same-seed ground-truth track is added as a line (no ghosts)")
    a = p.parse_args()

    spacing = dict(DEFAULT_SPACING)
    for s in a.spacing:
        k, v = s.split("="); spacing[float(k)] = float(v)
    overrides = dict(s.split("=") for s in a.seed)

    df = pd.read_csv(a.trials)
    bags = Path(a.bags).expanduser()
    mdir = Path(a.manifests).expanduser() if a.manifests else bags.parent / "manifests"
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    field = DockField()
    index = []
    for deg in a.degrade:
        for direction in a.directions:
            cell = f"d{int(round(deg * 100))}_{direction}"
            if cell in overrides:
                seed = int(overrides[cell])
                sub = df[(df.arm == a.arm) & np.isclose(df.degradation, deg)
                         & (df.direction == direction) & (df.seed == seed)]
                if sub.empty:
                    raise SystemExit(f"{cell}: seed {seed} not in trials.csv")
                row = sub.iloc[0]
            else:
                row = pick_trial(df, a.arm, deg, direction)
            tid = row.trial_id
            man_path = mdir / f"{tid}.json"
            man = json.loads(man_path.read_text()) if man_path.exists() else None
            print(f"{cell}: {tid}", flush=True)
            rec = extract(bags / tid, man, spacing[deg], a.track_step, field)
            rec.update({"cell": cell, "label": f"{int(round(deg*100))} %, {DIR_LABEL.get(direction, direction)}",
                        "trial_id": tid, "arm": a.arm, "seed": int(row.seed),
                        "degradation": float(deg), "direction": direction,
                        "scored": {k: (None if pd.isna(row.get(k)) else float(row.get(k)))
                                   for k in ("time_to_zone_s", "dist_to_zone_min_m", "energy_N2s")
                                   if k in row}})
            rec["baselines"] = {}
            for barm in a.baselines:
                bsub = df[(df.arm == barm) & np.isclose(df.degradation, deg)
                          & (df.direction == direction) & (df.seed == int(row.seed))]
                if bsub.empty:
                    print(f"   no {barm} trial with seed {int(row.seed)}; skipped"); continue
                btid = bsub.iloc[0].trial_id
                try:
                    b = extract_track(bags / btid, a.track_step, field); b["trial_id"] = btid
                    rec["baselines"][barm] = b
                    print(f"   {barm}: {btid} ({b['outcome']} at {b['t_end_since_fault']:.0f} s)")
                except Exception as e:
                    print(f"   {barm}: {btid} failed: {e}")
            (out / f"{cell}.json").write_text(json.dumps(rec, indent=1))
            preview(rec, out / f"{cell}_preview.png")
            print(f"   {rec['outcome']}, {len(rec['ghosts'])} ghosts, {len(rec['refs'])} references, "
                  f"terminal at {rec['t_end_since_fault']:.0f} s", flush=True)
            index.append({"cell": cell, "trial_id": tid, "outcome": rec["outcome"],
                          "n_ghosts": len(rec["ghosts"]), "n_refs": len(rec["refs"])})
    pd.DataFrame(index).to_csv(out / "index.csv", index=False)
    print(pd.DataFrame(index).to_string(index=False))


if __name__ == "__main__":
    main()
