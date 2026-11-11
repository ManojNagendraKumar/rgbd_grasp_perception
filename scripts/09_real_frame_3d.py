"""Run the full detect and locate pipeline on one real RGBD frame and compare with the annotated object positions.

The frame is the public example of GraspNet 1Billion (Fang et al., CVPR 2020): a RealSense camera on a robot arm.
Its metadata holds the camera intrinsics and the annotated 6D pose of every object. It has no camera to robot
transform, so the table plane is fitted first and used as the base frame (z up, table at z = 0), which is the frame
the 3D lifting expects. Nothing in the detector or the lifting is changed for real data.

The detector only knows five simulated classes and none of the real objects belong to them, so the class labels
are wrong or missing (see 08_sim_vs_real.py). The 3D step does not use the class, so each box is still located in
3D and compared with the annotated object whose position falls inside the box. Annotated positions are the origins
of the object models, which are close to, but not always exactly at, the object centre.

Output: results/real_frame_3d.png, results/real_frame_3d.json
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.geometry import backproject, invert, project, transform  # noqa: E402
from perception.lift3d import METHODS, PIPELINE_METHOD  # noqa: E402

URL = "https://raw.githubusercontent.com/graspnet/graspnet-baseline/main/doc/example_data/"
WEIGHTS = ROOT / "models/best.pt"
CONF, IMGSZ = 0.5, 512


def load_frame():
    dst = ROOT / "data/real_example"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("color.png", "depth.png", "workspace_mask.png", "meta.mat"):
        if not (dst / name).exists():
            urllib.request.urlretrieve(URL + name, dst / name)
    meta = sio.loadmat(dst / "meta.mat")
    rgb = np.array(Image.open(dst / "color.png").convert("RGB"))
    depth = np.array(Image.open(dst / "depth.png")).astype(np.float64) / float(meta["factor_depth"][0, 0])
    ws = np.array(Image.open(dst / "workspace_mask.png")) > 0
    return rgb, depth, ws, meta


def table_frame(depth, K, ws, rng, iters=400, thresh=0.005):
    """Fit the table plane with RANSAC and return T_table_cam (z up, origin on the table under the scene)."""
    v, u = np.nonzero(ws & (depth > 0))
    pts = backproject(K, u.astype(float), v.astype(float), depth[v, u])
    sample = pts[rng.choice(len(pts), min(30000, len(pts)), replace=False)]
    best, bn, bd = None, None, None
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(sample @ n - n @ a) < thresh
        if best is None or inl.sum() > best.sum():
            best, bn, bd = inl, n, -n @ a
    q = pts[np.abs(pts @ bn + bd) < thresh]
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    if n @ (-c) < 0:                                       # normal toward the camera, so z points up
        n = -n
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ c
    return T


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ultralytics import YOLO

    rgb, depth, ws, meta = load_frame()
    K = meta["intrinsic_matrix"].astype(float)
    T_table_cam = table_frame(depth, K, ws, np.random.default_rng(0))
    T_cam_table = invert(T_table_cam)
    gt_cam = meta["poses"][:, 3, :].T                       # annotated object positions, camera frame
    gt_tab = transform(T_table_cam, gt_cam)
    gt_px = project(K, gt_cam)

    r = YOLO(str(WEIGHTS)).predict(rgb[:, :, ::-1].copy(), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    lift = METHODS[PIPELINE_METHOD]
    rows = []
    for b, s, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy()):
        p = lift(depth, K, b, T_table_cam, table_z_base=0.0)
        inside = [i for i, (gu, gv) in enumerate(gt_px) if b[0] <= gu <= b[2] and b[1] <= gv <= b[3]]
        row = {"class_predicted": r.names[int(c)], "score": round(float(s), 3), "box": [round(float(v), 1) for v in b],
               "position_table_frame_m": None if p is None else [round(float(v), 4) for v in p],
               "annotated_object_inside_box": None, "error_mm": None}
        if p is not None and inside:
            i = min(inside, key=lambda i: np.linalg.norm(gt_tab[i] - p))
            e = gt_tab[i] - p
            row.update(annotated_object_inside_box=int(meta["cls_indexes"].ravel()[i]),
                       error_mm={"xyz": round(1000 * float(np.linalg.norm(e)), 1),
                                 "horizontal": round(1000 * float(np.linalg.norm(e[:2])), 1),
                                 "height": round(1000 * float(abs(e[2])), 1)})
        rows.append(row)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.16))
    ax[0].imshow(rgb)
    ax[0].axis("off")
    ax[0].scatter(gt_px[:, 0], gt_px[:, 1], marker="x", s=55, color="crimson", linewidths=2)
    for row in rows:
        x1, y1, x2, y2 = row["box"]
        ax[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=2))
        txt = row["class_predicted"]
        if row["error_mm"]:
            txt += f" {row['error_mm']['xyz']:.0f} mm"
        elif row["position_table_frame_m"]:
            txt += ": none inside"
        ax[0].text(x1 + 3, y2 - 6, txt, fontsize=8, bbox=dict(fc="yellow", ec="none", pad=1))
        if row["position_table_frame_m"]:
            uv = project(K, transform(T_cam_table, np.array([row["position_table_frame_m"]])))[0]
            ax[0].scatter(uv[0], uv[1], s=45, color="cyan", edgecolors="k", linewidths=0.8, zorder=5)
    ax[0].set_title("Real frame: boxes, 3D positions found (cyan) and annotated positions (red x)", fontsize=11)

    ax[1].scatter(gt_tab[:, 0], gt_tab[:, 1], marker="x", s=60, color="crimson", linewidths=2,
                  label="annotated object position")
    found = [row for row in rows if row["position_table_frame_m"]]
    for k, row in enumerate(found):
        p = row["position_table_frame_m"]
        ax[1].scatter(p[0], p[1], s=50, color="cyan", edgecolors="k", linewidths=0.8,
                      label="3D position found" if k == 0 else None, zorder=5)
        ax[1].annotate(row["class_predicted"], (p[0], p[1]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x in table frame (m)")
    ax[1].set_ylabel("y in table frame (m)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc="upper left", fontsize=8)
    matched = [row["error_mm"]["xyz"] for row in rows if row["error_mm"]]
    ax[1].set_title(f"Top view: {len(matched)} boxes matched to an annotated object,\n"
                    f"median 3D distance {np.median(matched):.0f} mm", fontsize=11)
    fig.savefig(ROOT / "results/real_frame_3d.png", dpi=110, bbox_inches="tight")

    out = {"lifting_method": PIPELINE_METHOD, "confidence_threshold": CONF,
           "boxes": rows, "matched_boxes": len(matched),
           "median_distance_mm": float(np.median(matched)) if matched else None,
           "notes": "Distances are to the annotated model origin of the object whose origin falls inside the box. "
                    "Model origins are near, but not always exactly at, the object centre, so treat the distances "
                    "as a rough check on one frame, not as a benchmark."}
    (ROOT / "results/real_frame_3d.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
