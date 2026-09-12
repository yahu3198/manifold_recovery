"""rosbag2 -> force-segment store, without a ROS 2 installation (rev 4.2).

Uses the pure-Python ``rosbags`` package. Two sources:
  (a) logging sessions named env_log_ssX_<direction>_runN, and
  (b) deployment-trial bags named <arm>_dNN_ssX_<direction>_seedNNNN
      (run_trials.py), of which only the PRE-FAULT part is used.

Topics: /wamv/disturbance (TwistStamped, body frame), /wamv/prediction_metrics
(Float64MultiArray; index 0 = prediction confidence in [0, 1]),
/wamv/sensors/position/ground_truth_odometry (heading filter),
/wamv/thruster_health (pre-fault mask, deployment bags).

Rev 4.2 corrections:
  * sigma_theta = kappa / max(sqrt(confidence), 0.3) - eps from index 0
    (previously the mean of the whole metrics array, which mixed a confidence,
    a frequency, forces and a covariance trace).
  * heading filter: a window is kept only if the heading stays within
    +-heading_tol_deg of heading_ref (default west, the recovery approach), so
    the body-frame forces in the store match the headings the certificate
    evaluates; eastbound return legs and turns are dropped.
  * pre-fault mask for trial bags.
  * windows are overlapping (stride = seg_len / 2) for more realisations.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .forecast import sigma_theta_from_confidence


def _parse_condition(name: str):
    m = re.search(r"ss(\d)_(beneficial|nominal)", name)
    if not m:
        raise ValueError(f"bag name '{name}' lacks ssX_<direction> tag")
    return int(m.group(1)), m.group(2)


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def read_bag(bag: Path):
    """Returns dict of time-stamped series (seconds) from one bag."""
    from rosbags.highlevel import AnyReader
    out = {"t_w": [], "w": [], "t_c": [], "conf": [], "t_o": [], "psi": [], "pos": [],
           "t_h": [], "h": []}
    want = {"/wamv/disturbance", "/wamv/prediction_metrics",
            "/wamv/sensors/position/ground_truth_odometry", "/wamv/thruster_health"}
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in want]
        for conn, ts, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype); t = ts * 1e-9
            if conn.topic == "/wamv/disturbance":
                out["t_w"].append(t); out["w"].append([m.twist.linear.x, m.twist.linear.y, m.twist.angular.z])
            elif conn.topic == "/wamv/prediction_metrics":
                if len(m.data): out["t_c"].append(t); out["conf"].append(float(m.data[0]))
            elif conn.topic == "/wamv/sensors/position/ground_truth_odometry":
                out["t_o"].append(t); out["psi"].append(_yaw(m.pose.pose.orientation))
                out["pos"].append([m.pose.pose.position.x, m.pose.pose.position.y])
            else:
                if len(m.data) >= 2: out["t_h"].append(t); out["h"].append([m.data[0], m.data[1]])
    return {k: np.asarray(v) for k, v in out.items()}


def extract_force_segments(bag_dirs, out_path, seg_len_s: float = 160.0,
                           resample_dt: float = 0.1, heading_ref_deg: float = 180.0,
                           heading_tol_deg: float = 45.0, kappa: float = 1.5,
                           eps: float = 0.01, verbose: bool = True):
    segs, sigmas, seas, dirns, srcs = [], [], [], [], []
    n_per = int(seg_len_s / resample_dt); stride = max(n_per // 2, 1)
    for bag in map(Path, bag_dirs):
        sea, dirn = _parse_condition(bag.name)
        d = read_bag(bag)
        if len(d["t_w"]) < 10:
            if verbose: print(f"  {bag.name}: no disturbance data, skipped")
            continue
        t0 = d["t_w"][0]
        tt = np.arange(0.0, d["t_w"][-1] - t0, resample_dt)
        Wr = np.stack([np.interp(tt, d["t_w"] - t0, d["w"][:, j]) for j in range(3)], axis=1)
        # heading and pre-fault masks on the resampled grid
        ok = np.ones(len(tt), bool)
        if len(d["t_o"]):
            psi = np.interp(tt, d["t_o"] - t0, np.unwrap(d["psi"]))
            ok &= np.abs(_wrap(psi - np.deg2rad(heading_ref_deg))) <= np.deg2rad(heading_tol_deg)
        if len(d["t_h"]):
            hmin = np.interp(tt, d["t_h"] - t0, d["h"].min(axis=1))
            ok &= hmin > 99.9
        conf = np.interp(tt, d["t_c"] - t0, d["conf"]) if len(d["t_c"]) else np.full(len(tt), 0.5)
        n_kept = 0
        for s0 in range(0, len(tt) - n_per + 1, stride):
            if not ok[s0:s0 + n_per].all():
                continue
            segs.append(Wr[s0:s0 + n_per]); n_kept += 1
            sigmas.append(sigma_theta_from_confidence(float(np.mean(conf[s0:s0 + n_per])), kappa, eps))
            seas.append(sea); dirns.append(dirn); srcs.append(bag.name)
        if verbose:
            print(f"  {bag.name}: {n_kept} windows of {seg_len_s:.0f}s (heading/pre-fault filtered)")
    if not segs:
        raise RuntimeError("no windows passed the filters")
    np.savez_compressed(out_path, w_body=np.asarray(segs), sigma_theta=np.asarray(sigmas),
                        dt=resample_dt, sea_state=np.asarray(seas),
                        direction=np.asarray(dirns, dtype="S"), source=np.asarray(srcs, dtype="S"))
    if verbose:
        for ss in sorted(set(seas)):
            for dd in sorted(set(dirns)):
                n = sum(1 for a, b in zip(seas, dirns) if a == ss and b == dd)
                if n: print(f"  SS{ss} {dd}: {n} windows")
    return len(segs)
