#!/usr/bin/env python3
"""Sidecar node: fault-time recovery proposal for the VRX deployment trials.

Runs beside wamv_mpc_node (ref_source: manifold). Watches /wamv/thruster_health
for the fault, snapshots the state, forecasts the environment from the last
60 s of /wamv/disturbance, runs the chosen arm, and publishes the selected
candidate on /wamv/manifold_ref in the layout WAMV_MPC::manifold_ref_cb
expects. Also publishes a proposal summary (/wamv/manifold_proposal), writes a
JSON log per trial, and optionally RViz markers of the certified candidates.

Arms:  manifold  decode from the trained CVAE (RecoveryPipeline.propose)
       b4        rejection sampling, same certificate and tail (propose_b4)

Run (venv must see the ROS 2 Python packages, e.g. created with
--system-site-packages, and ROS sourced):
    python ros2/vrx_manifold_node.py --arm manifold --ckpt runs/ckpt.pt \
        --log-dir ~/usv_ws/experiments/logs --trial-id ss3_beneficial_s0300
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy            # noqa: E402
from nav_msgs.msg import Odometry                              # noqa: E402
from geometry_msgs.msg import TwistStamped                     # noqa: E402
from std_msgs.msg import Float64MultiArray                     # noqa: E402

from manifold_recovery.config import load                      # noqa: E402
from manifold_recovery.io_bridge.forecast import ForceForecaster, sigma_theta_from_confidence  # noqa: E402
from manifold_recovery.io_bridge.reference import plan_to_rows, rows_to_message_data  # noqa: E402
from manifold_recovery.scenario import ZONES                   # noqa: E402


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class ManifoldSidecar(Node):
    def __init__(self, a):
        super().__init__("vrx_manifold_node")
        self.a = a
        self.cfg = load(a.config)
        self.pipe = None                 # built lazily (imports torch/casadi)
        self.fc = ForceForecaster(window_s=a.window_s, dt=0.1)
        self.state = None                # [x, y, psi, u, v, r]
        self.h = np.array([1.0, 1.0])
        self.conf = 0.5
        self.triggered = False
        self.done = False
        self.t_trigger = None
        self.rng = np.random.default_rng(a.seed)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/wamv/sensors/position/ground_truth_odometry",
                                 self.on_odom, qos)
        self.create_subscription(TwistStamped, "/wamv/disturbance", self.on_dist, qos)
        self.create_subscription(Float64MultiArray, "/wamv/thruster_health", self.on_health, qos)
        self.create_subscription(Float64MultiArray, "/wamv/prediction_metrics", self.on_pred, qos)
        self.ref_pub = self.create_publisher(Float64MultiArray, "/wamv/manifold_ref", 10)
        self.prop_pub = self.create_publisher(Float64MultiArray, "/wamv/manifold_proposal", 10)
        self.marker_pub = None
        if a.viz:
            from visualization_msgs.msg import MarkerArray
            self.marker_pub = self.create_publisher(MarkerArray, "/wamv/manifold_markers", 5)
        self.get_logger().info(f"sidecar ready: arm={a.arm} trial={a.trial_id}")
        self._build_pipeline()          # model and planners load now, not at fault time

    # ---- callbacks ---------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.state = np.array([m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quat(q),
                               m.twist.twist.linear.x, m.twist.twist.linear.y,
                               m.twist.twist.angular.z])

    def on_dist(self, m):
        self.fc.push(self.now(), [m.twist.linear.x, m.twist.linear.y, m.twist.angular.z])

    def on_pred(self, m):
        if len(m.data) > 0:
            self.conf = float(m.data[0])

    def on_health(self, m):
        if self.done or len(m.data) < 2:
            return
        h = np.clip(np.asarray(m.data[:2], float) / 100.0, 0.0, 1.0)
        self.h = h
        if not self.triggered and (h < 0.999).any():
            self.triggered = True
            self.t_trigger = self.now()
            self.get_logger().warn(f"fault detected: H={h.round(3).tolist()} at t={self.t_trigger:.2f}")
            self.handle_fault()

    # ---- the fault-time job ------------------------------------------------
    def _build_pipeline(self):
        if self.pipe is None:
            from manifold_recovery.pipeline.online import RecoveryPipeline
            ckpt = self.a.ckpt if self.a.arm == "manifold" else None
            self.pipe = RecoveryPipeline(ckpt, self.cfg, seed=self.a.seed)

    def handle_fault(self):
        if self.state is None or not self.fc.ready(min_s=self.a.min_history_s):
            self.get_logger().error("no state or too little disturbance history at the fault; nothing published")
            self._log({"error": "no_state_or_history"})
            self.done = True
            return
        x0 = self.state.copy()
        h1, h2 = float(self.h[0]), float(self.h[1])
        T_h, _ = self._horizon()
        w_hat, w_true, w_dt, fit = self.fc.forecast(T_h + 5.0, self.rng)
        sig = sigma_theta_from_confidence(self.conf, self.cfg.wrench.kappa, self.cfg.wrench.eps_conf)
        self._build_pipeline()
        t0 = time.perf_counter()
        if self.a.arm == "manifold":
            prop = self.pipe.propose(h1, h2, w_hat, sig, x0, w_true, w_dt)
        else:
            prop = self.pipe.propose_b4(h1, h2, w_hat, sig, x0, w_true, w_dt)
        wall = time.perf_counter() - t0
        log = {"trial_id": self.a.trial_id, "arm": self.a.arm, "t_trigger": self.t_trigger,
               "x0": x0.tolist(), "H": [h1, h2], "sigma_theta": sig, "confidence": self.conf,
               "forecast_fit": fit, "n_decoded": prop.n_decoded, "n_certified": prop.n_certified,
               "cert_breakdown": prop.cert_breakdown, "alpha_bar": prop.alpha_bar,
               "timings": prop.timings, "wall_total_s": wall,
               "candidates": [{"zone": c.zone_id, "signature": list(map(int, c.signature)),
                               "plan_cost": c.plan_cost, "rollout_passed": bool(c.cert.passed),
                               "rollout_reason": c.cert.reason,
                               "plan_feasible": bool(c.timings.get("plan_feasible", False))}
                              for c in prop.candidates]}
        chosen = next((c for c in prop.candidates if c.plan is not None), None)
        if chosen is None:
            self.get_logger().error(f"no candidate to publish (certified {prop.n_certified}/{prop.n_decoded}); "
                                    f"MPC will fall back after its timeout")
            log["published"] = False
            self._log(log); self._publish_summary(prop, None, wall); self.done = True
            return
        rows = plan_to_rows(chosen.plan, T_h, 0.05, feedforward=self.a.feedforward)
        msg = Float64MultiArray()
        msg.data = rows_to_message_data(rows, self.t_trigger, 0.05)
        self.ref_pub.publish(msg)
        log.update({"published": True, "chosen_zone": chosen.zone_id,
                    "chosen_signature": list(map(int, chosen.signature)),
                    "chosen_plan_cost": chosen.plan_cost, "rows": int(len(rows)),
                    "publish_latency_s": self.now() - self.t_trigger})
        self.get_logger().info(f"published {len(rows)} rows -> {ZONES[chosen.zone_id].name} "
                               f"(class {chosen.signature}), latency {log['publish_latency_s']:.2f}s, "
                               f"certified {prop.n_certified}/{prop.n_decoded}")
        self._log(log); self._publish_summary(prop, chosen, wall)
        if self.marker_pub is not None:
            self._publish_markers(prop, chosen)
        self.done = True

    # ---- helpers -----------------------------------------------------------
    def _horizon(self):
        from manifold_recovery.traj.rtp import RTP
        return RTP(self.cfg.trajectory).horizon()

    def _publish_summary(self, prop, chosen, wall):
        m = Float64MultiArray()
        m.data = [float(prop.n_decoded), float(prop.n_certified), float(len(prop.candidates)),
                  float(chosen.zone_id) if chosen else -1.0,
                  float(chosen.plan_cost) if chosen else float("nan"),
                  float(prop.timings.get("decode", 0.0)), float(prop.timings.get("certify", 0.0)),
                  float(prop.timings.get("finetune_and_rollout", 0.0)), float(wall)]
        self.prop_pub.publish(m)

    def _log(self, d):
        out = Path(self.a.log_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.a.trial_id}_{self.a.arm}.json").write_text(json.dumps(d, indent=2, default=float))

    def _publish_markers(self, prop, chosen):
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        arr = MarkerArray()
        for i, c in enumerate(prop.candidates):
            mk = Marker(); mk.header.frame_id = "map"; mk.ns = "manifold"; mk.id = i
            mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD; mk.scale.x = 0.6
            mk.color.a = 1.0
            if c is chosen:
                mk.color.g = 0.8
            else:
                mk.color.b = 0.9; mk.color.g = 0.4
            mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.3) for p in c.xi]
            arr.markers.append(mk)
        self.marker_pub.publish(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["manifold", "b4"], default="manifold")
    p.add_argument("--config", default=str(ROOT / "configs/spike.yaml"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/ckpt.pt"))
    p.add_argument("--log-dir", default="~/usv_ws/experiments/logs")
    p.add_argument("--trial-id", default="manual")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--min-history-s", type=float, default=20.0)
    p.add_argument("--feedforward", action="store_true", help="fill Tp, Ts from the plan (default zeros)")
    p.add_argument("--viz", action="store_true", help="publish RViz markers of the candidates")
    a, ros_args = p.parse_known_args()
    rclpy.init(args=ros_args)
    node = ManifoldSidecar(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
