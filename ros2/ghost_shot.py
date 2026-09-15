#!/usr/bin/env python3
"""Open the trial world in Gazebo, spawn the ghost scene, set a fixed top-down
camera, pause, and screenshot. One run per panel.

    python ros2/ghost_shot.py ~/usv_ws/experiments/figure_ghosts/d50_beneficial.json

Reads <cell>.json (world path and name) and <cell>_ghosts.sdf next to it. The
camera pose is identical for every panel unless overridden, so the six panels
share one scale. Uses only the `gz` CLI (no ROS): --gz ign for Fortress.

Everything runs through gz services:
  /world/<name>/create      EntityFactory   spawn the ghost model
  /gui/move_to/pose         GUICamera       top-down camera
  /world/<name>/control     WorldControl    pause
  /gui/screenshot           StringMsg       save PNG (data = a directory; the GUI names
                                            the file by timestamp, the script renames it)
If the screenshot service is absent, the window stays open: use the camera icon
in the GUI or a desktop screenshot, then press Enter here to close.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def quat_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,        # x
            cr * sp * cy + sr * cp * sy,        # y
            cr * cp * sy - sr * sp * cy,        # z
            cr * cp * cy + sr * sp * sy)        # w


def gz_service(gz, name, reqtype, reptype, req, timeout=5000):
    cmd = [gz, "service", "-s", name, "--reqtype", f"gz.msgs.{reqtype}", "--reptype",
           f"gz.msgs.{reptype}", "--timeout", str(timeout), "--req", req]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and "true" in r.stdout.lower()
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f": {r.stdout.strip()} {r.stderr.strip()}"))
    return ok


def wait_for_service(gz, name, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run([gz, "service", "-l"], capture_output=True, text=True)
        if name in r.stdout:
            return True
        time.sleep(1.0)
    return False


def resource_env(extra: list[str]):
    env = os.environ.copy()
    paths = [p for p in env.get("GZ_SIM_RESOURCE_PATH", "").split(":") if p]
    try:
        pre = subprocess.run(["ros2", "pkg", "prefix", "wamv_description"], capture_output=True,
                             text=True, check=True).stdout.strip()
        paths.append(str(Path(pre) / "share" / "wamv_description" / "models"))
        paths.append(str(Path(pre) / "share" / "wamv_description"))
    except Exception:
        pass
    paths += extra
    env["GZ_SIM_RESOURCE_PATH"] = ":".join(dict.fromkeys(paths))
    env["IGN_GAZEBO_RESOURCE_PATH"] = env["GZ_SIM_RESOURCE_PATH"]
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cell_json")
    p.add_argument("--gz", default="gz", help="gz (Garden/Harmonic) or ign (Fortress)")
    p.add_argument("--world", default=None, help="override the trial world .sdf")
    p.add_argument("--sdf", default=None, help="override the ghost sdf (default <cell>_ghosts.sdf)")
    p.add_argument("--out", default=None, help="screenshot path (default <cell>.png next to the json)")
    p.add_argument("--cam", nargs=3, type=float, default=[-528.0, 222.0, 96.0],
                   help="camera x y z; z sets the scale. Same for all six panels.")
    p.add_argument("--cam-rpy", nargs=3, type=float, default=[0.0, math.pi / 2, math.pi / 2],
                   help="roll pitch yaw; pitch pi/2 looks straight down, yaw pi/2 puts +y (north) at the top")
    p.add_argument("--fit", action="store_true", help="centre the camera on the ghosts of this cell instead")
    p.add_argument("--resource-path", nargs="*", default=[], help="extra GZ_SIM_RESOURCE_PATH entries")
    p.add_argument("--settle-s", type=float, default=20.0)
    p.add_argument("--hold-s", type=float, default=6.0, help="seconds between camera move and screenshot")
    p.add_argument("--keep", action="store_true", help="leave Gazebo open after the screenshot")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cj = Path(a.cell_json).expanduser()
    cell = json.loads(cj.read_text())
    world = Path(a.world or cell["world"]).expanduser()
    wname = cell.get("world_name") or "default"
    sdf = Path(a.sdf or cj.with_name(cj.stem + "_ghosts.sdf")).expanduser()
    out = Path(a.out or cj.with_suffix(".png")).expanduser()
    if not sdf.exists():
        sys.exit(f"missing {sdf}; run 06_ghost_sdf.py first")

    cx, cy, cz = a.cam
    if a.fit:
        xs = [g["x"] for g in cell["ghosts"]]; ys = [g["y"] for g in cell["ghosts"]]
        cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    qx, qy, qz, qw = quat_rpy(*a.cam_rpy)
    cam_req = (f"pose: {{position: {{x: {cx}, y: {cy}, z: {cz}}}, "
               f"orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}}}")
    spawn_req = f'sdf_filename: "{sdf}", name: "ghosts_{cell["cell"]}"'

    print(f"cell {cell['cell']}  trial {cell['trial_id']}  world {world.name} ({wname})")
    print(f"camera at ({cx:.1f}, {cy:.1f}, {cz:.1f}); ghosts span x {min(g['x'] for g in cell['ghosts']):.0f}.."
          f"{max(g['x'] for g in cell['ghosts']):.0f}, y {min(g['y'] for g in cell['ghosts']):.0f}.."
          f"{max(g['y'] for g in cell['ghosts']):.0f}")
    if a.dry_run:
        print(f"{a.gz} sim -r {world}")
        print(f"{a.gz} service -s /world/{wname}/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --req '{spawn_req}'")
        print(f"{a.gz} service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --req '{cam_req}'")
        print(f"{a.gz} service -s /world/{wname}/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --req 'pause: true'")
        print(f"{a.gz} service -s /gui/screenshot --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --req 'data: \"{out.parent}/.shots\"'  # then rename the timestamped png")
        return

    env = resource_env(a.resource_path)
    gzp = subprocess.Popen([a.gz, "sim", "-r", str(world)], env=env, start_new_session=True)
    try:
        if not wait_for_service(a.gz, f"/world/{wname}/create", a.settle_s + 60):
            sys.exit(f"/world/{wname}/create never appeared; check the <world name> in {world}")
        time.sleep(a.settle_s)                       # let the wave field and GUI come up
        gz_service(a.gz, f"/world/{wname}/create", "EntityFactory", "Boolean", spawn_req)
        time.sleep(2.0)
        gz_service(a.gz, "/gui/move_to/pose", "GUICamera", "Boolean", cam_req)
        time.sleep(a.hold_s)
        gz_service(a.gz, f"/world/{wname}/control", "WorldControl", "Boolean", "pause: true")
        time.sleep(1.0)
        # /gui/screenshot treats `data` as a directory and writes <timestamp>.png inside it
        shot_dir = out.parent / ".shots"; shot_dir.mkdir(parents=True, exist_ok=True)
        before = set(shot_dir.glob("*.png"))
        shot = gz_service(a.gz, "/gui/screenshot", "StringMsg", "Boolean", f'data: "{shot_dir}"')
        if shot:
            time.sleep(2.0)
            new = sorted(set(shot_dir.glob("*.png")) - before, key=lambda f: f.stat().st_mtime)
            if new:
                new[-1].replace(out); print(f"screenshot -> {out}")
            else:
                shot = False; print(f"screenshot service returned true but wrote nothing in {shot_dir}")
        else:
            print("screenshot service unavailable: take it from the GUI (camera icon) or the desktop.")
        if a.keep or not shot:
            input("Gazebo stays open; adjust the camera if needed, then press Enter to close... ")
    finally:
        try:
            os.killpg(gzp.pid, signal.SIGINT); time.sleep(3.0)
            if gzp.poll() is None:
                os.killpg(gzp.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    main()
