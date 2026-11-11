"""Render short video sequences with ground truth identities for tracking evaluation.

Three conditions, four sequences each (80 frames, fixed camera and lighting per sequence):
  crossing  objects slide across the table and pass in front of each other
  occluder  the same, plus a worker's arm (a skin coloured capsule) sweeping through the scene
  fast      objects move about three times faster, so boxes barely overlap between frames

Output: data/tracking/<condition>_<k>/frame_XXXX.png and gt.json (per frame: id, class, box).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.sim_scene import CLASS_NAMES, WORKSPACE_X, WORKSPACE_Y, Workcell  # noqa: E402

N_FRAMES = 80
MIN_VISIBLE_PX = 60


def make_arm(wc):
    col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.045, height=0.45, physicsClientId=wc.cid)
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.045, length=0.45, rgbaColor=[0.87, 0.67, 0.55, 1],
                              physicsClientId=wc.cid)
    quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # lying horizontally, pointing along y
    return p.createMultiBody(0, col, vis, [5, 5, 5], quat, physicsClientId=wc.cid), quat


def run_sequence(name, condition, seed, out_root):
    wc = Workcell(seed=seed)
    rng = wc.rng
    wc.spawn_random_objects(4, 6)
    wc.randomize_appearance()
    wc.randomize_camera(jitter_pos=0.04, jitter_target=0.03)
    light = wc.sample_lighting()
    speed = (0.012, 0.022) if condition == "fast" else (0.003, 0.007)   # metres per frame

    movers = {}
    for uid in wc.objects:
        pos, quat = p.getBasePositionAndOrientation(uid, physicsClientId=wc.cid)
        target = np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y)])
        d = target - np.array(pos[:2])
        v = d / (np.linalg.norm(d) + 1e-9) * rng.uniform(*speed)
        movers[uid] = {"xy": np.array(pos[:2]), "z": pos[2], "quat": quat, "v": v}

    arm = None
    if condition == "occluder":
        arm, arm_quat = make_arm(wc)
        arm_x = rng.uniform(0.48, 0.66)
        arm_start = int(rng.integers(15, 35))

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    gt = []
    for t in range(N_FRAMES):
        for uid, m in movers.items():
            nxt = m["xy"] + m["v"]
            # bounce at the workspace edge so objects stay on the table
            for k, (lo, hi) in enumerate((WORKSPACE_X, WORKSPACE_Y)):
                if not lo <= nxt[k] <= hi:
                    m["v"][k] *= -1
            m["xy"] = m["xy"] + m["v"]
            p.resetBasePositionAndOrientation(uid, [*m["xy"], m["z"]], m["quat"], physicsClientId=wc.cid)
        if arm is not None:
            s = (t - arm_start) / 30.0                   # sweep across the table in 30 frames
            y = -0.55 + 1.1 * s if 0 <= s <= 1 else 5.0
            p.resetBasePositionAndOrientation(arm, [arm_x, y, wc.table_top_z + 0.14], arm_quat,
                                              physicsClientId=wc.cid)
        f = wc.render(lighting=light)
        cv2.imwrite(str(out / f"frame_{t:04d}.png"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
        boxes = []
        for uid, cls in wc.objects.items():
            msk = f.seg == uid
            if msk.sum() < MIN_VISIBLE_PX:
                continue
            ys, xs = np.nonzero(msk)
            boxes.append({"id": int(uid), "cls": CLASS_NAMES.index(cls),
                          "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
        gt.append(boxes)
    (out / "gt.json").write_text(json.dumps({"condition": condition, "frames": gt}))
    wc.close()


def main():
    out_root = ROOT / "data/tracking"
    seed = 700000
    for condition in ("crossing", "occluder", "fast"):
        for k in range(4):
            seed += 1
            run_sequence(f"{condition}_{k}", condition, seed, out_root)
            print("done", condition, k, flush=True)


if __name__ == "__main__":
    main()
