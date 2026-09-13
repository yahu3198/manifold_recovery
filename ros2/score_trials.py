#!/usr/bin/env python3
"""Offline scorer for the deployment trials (from bags, not from the node).

Per trial: fault time t_f (first /wamv/thruster_health < 100), success = the
ground-truth position enters ANY zone polygon within --post-fault-s of t_f,
collision = minimum signed distance to any obstacle (docks + shoreline,
manifold_recovery.score.obstacles.DockField, the same geometry the
certificate uses) < 0 after t_f, time to zone, thrust energy after t_f
(sum (Tp^2 + Ts^2) dt, plus the R(H)-weighted effort), sidecar latency and
fallback from /wamv/manifold_status. Writes trials.csv and a summary table by
(arm, degradation, direction), and optionally builds the logged-force store
from the pre-fault segments of all bags.

    python ros2/score_trials.py --bags ~/usv_ws/experiments/bags --out ~/usv_ws/experiments \
        --build-store data_out/force_segments.npz
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manifold_recovery.scenario import zone_of                        # noqa: E402
from manifold_recovery.score.obstacles import DockField               # noqa: E402
from manifold_recovery.io_bridge.bag_reader import read_bag, extract_force_segments  # noqa: E402

PAT = re.compile(r"(?P<arm>[a-z0-9]+)_d(?P<deg>\d+)_ss(?P<ss>\d)_(?P<dir>beneficial|nominal)_seed(?P<seed>\d+)")


def score_one(bag: Path, post_fault_s: float, field: DockField, R0=0.001, gamma=0.0075):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    d = read_bag(bag)
    st = {"t_thr": [], "Tp": [], "Ts": [], "t_ms": [], "ms": []}
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = [c for c in reader.connections if c.topic in
                 ("/wamv/thrusters/left/thrust", "/wamv/thrusters/right/thrust", "/wamv/manifold_status")]
        for conn, ts, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype); t = ts * 1e-9
            if conn.topic.endswith("left/thrust"): st["t_thr"].append(t); st["Tp"].append(float(m.data))
            elif conn.topic.endswith("right/thrust"): st["Ts"].append(float(m.data))
            else: st["t_ms"].append(t); st["ms"].append(list(m.data))
    if len(d["t_h"]) == 0 or len(d["t_o"]) == 0:
        return {"error": "missing health or odometry"}
    fault_idx = np.flatnonzero(d["h"].min(axis=1) < 99.9)
    if len(fault_idx) == 0:
        return {"error": "no fault in bag"}
    t_f = float(d["t_h"][fault_idx[0]])
    h_after = d["h"][fault_idx[0]] / 100.0
    sel = (d["t_o"] >= t_f) & (d["t_o"] <= t_f + post_fault_s)
    pos = d["pos"][sel]; t_o = d["t_o"][sel]
    z = zone_of(pos)
    hit = np.flatnonzero(z >= 0)
    sd = field.signed_distance(pos)
    t_arr = float(t_o[hit[0]] - t_f) if len(hit) else np.nan
    collided = bool((sd < 0).any())
    # energy after the fault
    n = min(len(st["Tp"]), len(st["Ts"]), len(st["t_thr"]))
    Tt = np.asarray(st["t_thr"][:n]); Tp = np.asarray(st["Tp"][:n]); Ts = np.asarray(st["Ts"][:n])
    m = (Tt >= t_f) & (Tt <= t_f + (t_arr if np.isfinite(t_arr) else post_fault_s))
    dt = np.median(np.diff(Tt)) if n > 2 else 0.05
    wu = np.array([R0 + gamma * (1 - h_after[0]), R0 + gamma * (1 - h_after[1])])
    e_raw = float(((Tp[m] ** 2 + Ts[m] ** 2) * dt).sum())
    e_w = float(((wu[0] * Tp[m] ** 2 + wu[1] * Ts[m] ** 2) * dt).sum())
    lat, fb = np.nan, False
    if st["ms"]:
        ms = np.asarray(st["ms"])
        if ms.shape[1] >= 4:
            v = ms[:, 2][ms[:, 2] >= 0]; lat = float(v[0]) if len(v) else np.nan
            fb = bool((ms[:, 3] > 0.5).any())
    return {"t_fault": t_f, "h1": float(h_after[0]), "h2": float(h_after[1]),
            "success": bool(len(hit)) and not collided, "arrived": bool(len(hit)),
            "zone": int(z[hit[0]]) if len(hit) else -1, "time_to_zone_s": t_arr,
            "collided": collided, "min_clearance_m": float(sd.min()) if len(sd) else np.nan,
            "energy_N2s": e_raw, "effort_weighted": e_w, "ref_latency_s": lat, "fallback": fb,
            "final_x": float(pos[-1, 0]) if len(pos) else np.nan,
            "final_y": float(pos[-1, 1]) if len(pos) else np.nan}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bags", required=True)
    p.add_argument("--out", default=".")
    p.add_argument("--post-fault-s", type=float, default=180.0)
    p.add_argument("--build-store", default=None, help="also build the force store (npz) from pre-fault segments")
    a = p.parse_args()
    field = DockField()
    rows = []
    bags = sorted(pp for pp in Path(a.bags).expanduser().iterdir() if pp.is_dir())
    for bag in bags:
        mm = PAT.search(bag.name)
        if not mm:
            print(f"skip {bag.name} (name pattern)"); continue
        r = {"trial_id": bag.name, "arm": mm["arm"], "degradation": int(mm["deg"]) / 100.0,
             "sea_state": int(mm["ss"]), "direction": mm["dir"], "seed": int(mm["seed"])}
        try:
            r.update(score_one(bag, a.post_fault_s, field))
        except Exception as e:      # keep going through a campaign
            r["error"] = repr(e)
        rows.append(r); print(f"{bag.name}: {r.get('success', r.get('error'))} "
                              f"t={r.get('time_to_zone_s', float('nan')):.0f}s clr={r.get('min_clearance_m', float('nan')):.1f}")
    df = pd.DataFrame(rows)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trials.csv", index=False)
    ok = df[df.get("error").isna()] if "error" in df else df
    if len(ok):
        g = ok.groupby(["arm", "degradation", "direction"]).agg(
            n=("success", "size"), success_rate=("success", "mean"), collided=("collided", "mean"),
            fallback=("fallback", "mean"), time_to_zone_s=("time_to_zone_s", "median"),
            energy_N2s=("energy_N2s", "median"), latency_s=("ref_latency_s", "median"))
        g.to_csv(out / "summary.csv")
        print("\n" + g.to_string(float_format=lambda v: f"{v:.2f}"))
    if a.build_store:
        print("\nbuilding force store from pre-fault segments")
        n = extract_force_segments(bags, a.build_store)
        print(f"{n} windows -> {a.build_store}")


if __name__ == "__main__":
    main()
