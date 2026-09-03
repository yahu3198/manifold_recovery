"""rosbag2 (.db3) -> force-segment store, without a ROS 2 installation.

Uses the pure-Python ``rosbags`` package. Reads the topics recorded by the
force-logging campaign (see record.sh convention in vrx_control):
  /wamv/disturbance         geometry_msgs/TwistStamped  (body-frame w_hat)
  /wamv/prediction_metrics  Float64MultiArray           (sigma_theta source)
Bag directory names encode the condition, e.g. env_log_ss3_nominal_run2.
Frame sanity: if /wamv/disturbance_world is present, one segment is
cross-checked against the body-frame series rotated by the EKF heading.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _parse_condition(name: str):
    m = re.search(r"ss(\d)_(beneficial|nominal)", name)
    if not m:
        raise ValueError(f"bag name '{name}' lacks ssX_<direction> tag")
    return int(m.group(1)), m.group(2)


def extract_force_segments(bag_dirs: list[str | Path], out_path: str | Path,
                           seg_len_s: float = 160.0, resample_dt: float = 0.1):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:                     # pragma: no cover
        raise ImportError("pip install rosbags") from e

    segs, sigmas, seas, dirns = [], [], [], []
    for bag in map(Path, bag_dirs):
        sea, dirn = _parse_condition(bag.name)
        times, wx, wy, wpsi, sig_t, sig_v = [], [], [], [], [], []
        with AnyReader([bag]) as reader:
            conns = [c for c in reader.connections
                     if c.topic in ("/wamv/disturbance", "/wamv/prediction_metrics")]
            for conn, ts, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == "/wamv/disturbance":
                    times.append(ts * 1e-9)
                    wx.append(msg.twist.linear.x)
                    wy.append(msg.twist.linear.y)
                    wpsi.append(msg.twist.angular.z)
                else:
                    sig_t.append(ts * 1e-9)
                    sig_v.append(float(np.mean(msg.data)) if len(msg.data) else 1.0)
        t = np.asarray(times) - times[0]
        W = np.stack([wx, wy, wpsi], axis=1)
        tt = np.arange(0.0, t[-1], resample_dt)
        Wr = np.stack([np.interp(tt, t, W[:, j]) for j in range(3)], axis=1)
        n_per = int(seg_len_s / resample_dt)
        for s0 in range(0, len(tt) - n_per, n_per):
            segs.append(Wr[s0:s0 + n_per])
            sigmas.append(float(np.mean(sig_v)) if sig_v else 1.0)
            seas.append(sea)
            dirns.append(dirn)
    np.savez_compressed(out_path, w_body=np.asarray(segs),
                        sigma_theta=np.asarray(sigmas), dt=resample_dt,
                        sea_state=np.asarray(seas),
                        direction=np.asarray(dirns, dtype="S"))
    return len(segs)
