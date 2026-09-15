#!/usr/bin/env python3
"""Build the ghost scene SDF for one cell from <cell>.json (06_ghost_poses.py).

One static model, no collision elements, no plugins:
  * one link per ghost with the WAM-V hull mesh (visual only), transparency
    fading from --alpha-first at the fault to opaque at the terminal instant
  * the ground-truth track as a chain of thin boxes (blue)
  * every published reference as a chain of thin boxes (red; replans lighter)
  * optional flat label tiles are not included: add "50 %, aligned" etc. in the
    figure layout, not in Gazebo.

    python scripts/06_ghost_sdf.py ~/usv_ws/experiments/figure_ghosts/d50_beneficial.json

Mesh URIs: the VRX fork keeps the meshes as separate models under
wamv_description/models/{WAM-V-Base,engine,propeller}/mesh/, and the package's
environment hook prepends that directory to GZ_SIM_RESOURCE_PATH, so with the
workspace sourced model://WAM-V-Base/mesh/WAM-V-Base.dae resolves.
--refs first+last draws only the stage-1 reference and the final replan.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TRACK_RGBA = (0.10, 0.40, 0.95, 1.0)      # ground truth
REF_RGBA = (0.90, 0.10, 0.10, 1.0)        # stage-1 reference
REPLAN_RGBA = (1.00, 0.55, 0.15, 1.0)     # replans


def _pose(x, y, z, yaw):
    return f"{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.5f}"


def _material(rgba, transparency=0.0):
    r, g, b, a = rgba
    return (f"<material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular><emissive>{0.4*r} {0.4*g} {0.4*b} 1</emissive></material>"
            + (f"<transparency>{transparency:.3f}</transparency>" if transparency > 0 else ""))


def polyline_link(name: str, xy: np.ndarray, rgba, width: float, z: float, step: float) -> str:
    """Chain of boxes along a polyline, one per segment after resampling to `step` m."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return ""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    if s[-1] < 1e-3:
        return ""
    si = np.append(np.arange(0.0, s[-1], step), s[-1])
    P = np.column_stack([np.interp(si, s, xy[:, 0]), np.interp(si, s, xy[:, 1])])
    vis = []
    for k in range(len(P) - 1):
        d = P[k + 1] - P[k]; L = float(np.hypot(*d))
        if L < 1e-3:
            continue
        mid = 0.5 * (P[k] + P[k + 1]); yaw = math.atan2(d[1], d[0])
        vis.append(f'<visual name="{name}_{k}"><pose>{_pose(mid[0], mid[1], z, yaw)}</pose>'
                   f"<geometry><box><size>{L + 0.6*width:.3f} {width:.3f} 0.08</size></box></geometry>"
                   f"{_material(rgba)}<cast_shadows>false</cast_shadows></visual>")
    return f'<link name="{name}"><pose>0 0 0 0 0 0</pose>{"".join(vis)}</link>'


def hull_material(mesh: str) -> str:
    """VRX colours the hull through PBR texture maps set in wamv_base.urdf.xacro,
    not in the DAE, so a bare mesh renders grey. The maps sit next to the mesh."""
    base = mesh.rsplit("/", 1)[0]
    return ("<material><diffuse>1 1 1 1</diffuse><specular>1 1 1 1</specular><pbr><metal>"
            f"<albedo_map>{base}/WAM-V_Albedo.png</albedo_map>"
            f"<normal_map>{base}/WAM-V_Normal.png</normal_map>"
            f"<roughness_map>{base}/WAM-V_Roughness.png</roughness_map>"
            f"<metalness_map>{base}/WAM-V_Metalness.png</metalness_map>"
            "</metal></pbr></material>")


def ghost_link(i: int, g: dict, mesh: str, engines: list[str], z: float, transparency: float,
               mesh_pose: str, textured: bool = True) -> str:
    vis = (f'<visual name="hull"><pose>{mesh_pose}</pose><geometry><mesh><uri>{mesh}</uri></mesh></geometry>'
           f"{hull_material(mesh) if textured else ''}"
           f"<transparency>{transparency:.3f}</transparency><cast_shadows>false</cast_shadows></visual>")
    for j, (uri, pose) in enumerate(engines):
        vis += (f'<visual name="engine_{j}"><pose>{pose}</pose><geometry><mesh><uri>{uri}</uri></mesh></geometry>'
                f"<transparency>{transparency:.3f}</transparency><cast_shadows>false</cast_shadows></visual>")
    tag = f' <!-- {g.get("tag", "")} t={g["t_since_fault"]:.0f}s -->' if g.get("tag") else ""
    return f'<link name="ghost_{i}">{tag}<pose>{_pose(g["x"], g["y"], z, g["psi"])}</pose>{vis}</link>'


def build(cell: dict, a) -> str:
    n = len(cell["ghosts"])
    links = []
    for i, g in enumerate(cell["ghosts"]):
        f = i / max(n - 1, 1)                                   # 0 at fault, 1 at terminal
        tr = a.alpha_first * (1.0 - f) if a.fade else 0.0
        engines = []
        if a.engine:
            # VRX engine visuals sit at +-1.03 m athwart, 2.37 m aft of the hull origin
            engines = [(a.engine, "-2.373776 1.027135 0.318237 0 0 0"),
                       (a.engine, "-2.373776 -1.027135 0.318237 0 0 0")]
        links.append(ghost_link(i, g, a.mesh, engines, a.z_ghost, tr, a.mesh_pose, not a.no_texture))
    if not a.no_track:
        links.append(polyline_link("track", cell["track"], TRACK_RGBA, a.line_width, a.z_line, a.line_step))
    if a.refs != "none":
        refs = list(enumerate(cell["refs"]))
        if a.refs == "first":
            refs = refs[:1]
        elif a.refs == "first+last" and len(refs) > 2:
            refs = [refs[0], refs[-1]]
        for k, r in refs:
            rgba = REF_RGBA if k == 0 else REPLAN_RGBA
            links.append(polyline_link(f"ref_{k}", r["xy"], rgba, 0.7 * a.line_width, a.z_line + 0.02, a.line_step))
    name = f"ghosts_{cell['cell']}"
    body = "\n".join(l for l in links if l)
    return (f'<?xml version="1.0"?>\n<sdf version="1.8">\n<model name="{name}">\n'
            f"<static>true</static>\n<self_collide>false</self_collide>\n{body}\n</model>\n</sdf>\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cell_json", nargs="+")
    p.add_argument("--mesh", default="model://WAM-V-Base/mesh/WAM-V-Base.dae")
    p.add_argument("--engine", default="model://engine/mesh/engine.dae", help="engine mesh URI; '' to omit")
    p.add_argument("--mesh-pose", default="0 0 0 0 0 0", help="hull mesh offset inside the ghost link")
    p.add_argument("--z-ghost", type=float, default=0.0, help="hull height above the water plane")
    p.add_argument("--z-line", type=float, default=0.25, help="track/reference height above the water plane")
    p.add_argument("--line-width", type=float, default=0.6)
    p.add_argument("--line-step", type=float, default=2.0)
    p.add_argument("--alpha-first", type=float, default=0.35,
                   help="transparency of the fault-time ghost (0 opaque, 1 invisible); later ghosts fade to 0")
    p.add_argument("--no-texture", action="store_true", help="skip the PBR hull textures")
    p.add_argument("--no-fade", dest="fade", action="store_false")
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--refs", default="all", choices=["all", "first", "first+last", "none"],
                   help="which published references to draw")
    a = p.parse_args()
    for cj in a.cell_json:
        cj = Path(cj).expanduser(); cell = json.loads(cj.read_text())
        out = cj.with_name(cj.stem + "_ghosts.sdf")
        out.write_text(build(cell, a))
        print(f"{out}: {len(cell['ghosts'])} ghosts, {len(cell['refs'])} references")


if __name__ == "__main__":
    main()
