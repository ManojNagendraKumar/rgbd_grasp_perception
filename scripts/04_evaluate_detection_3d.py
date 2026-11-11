"""Evaluate the detector (mAP on held out test scenes) and the 3D lifting into the robot base frame.

3D ground truth is each object's bounding box centre from the simulator. The camera pose used for lifting is
the TRUE pose corrupted by the error of the 15 pose hand eye calibration (results/handeye_calibration.npz),
so the reported 3D error includes calibration error, as it would on a real robot.

Writes results/detection_metrics.json, results/lifting_3d.json and two figures.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.geometry import invert, transform  # noqa: E402
from perception.lift3d import METHODS, PIPELINE_METHOD  # noqa: E402
from perception.sim_scene import CLASS_NAMES  # noqa: E402
from perception.tracker import iou_matrix  # noqa: E402


def match(pred_boxes, pred_cls, gt_boxes, gt_cls, thr=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    iou = iou_matrix(pred_boxes, gt_boxes)
    iou[pred_cls[:, None] != gt_cls[None, :]] = 0.0
    r, c = linear_sum_assignment(-iou)
    return [(i, j) for i, j in zip(r, c) if iou[i, j] >= thr]


def stats(errs):
    e = np.asarray(errs) * 1000.0
    return {"n": int(e.size), "median_mm": float(np.median(e)), "mean_mm": float(e.mean()),
            "p90_mm": float(np.percentile(e, 90))}


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()
    model = YOLO(args.weights)

    # ---------------- A) detection quality on the held out test split
    m = model.val(data=str(ROOT / "data/dataset.yaml"), split="test", imgsz=640, batch=8, device="cpu",
                  plots=False, verbose=False)
    det = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
           "recall": float(m.box.mr),
           "per_class_mAP50_95": {CLASS_NAMES[i]: float(v) for i, v in enumerate(m.box.maps)}}
    (ROOT / "results/detection_metrics.json").write_text(json.dumps(det, indent=2))
    print(json.dumps(det, indent=2))

    # ---------------- B) 3D lifting into the robot base frame
    he = np.load(ROOT / "results/handeye_calibration.npz")
    E = he["T_base_cam_est"] @ invert(he["T_base_cam_true"])   # calibration error as a rigid transform
    errs = {k: {"xyz": [], "xy": []} for k in list(METHODS) + ["pipeline_true_extrinsics", "pipeline_gt_boxes"]}
    examples = []
    files = sorted((ROOT / "data/test_rgbd").glob("*.npz"))
    for f in files:
        d = np.load(f)
        name = f.stem
        img = cv2.imread(str(ROOT / "data/yolo/images/test" / f"{name}.png"))
        T_base_world = invert(d["T_world_base"])
        T_base_cam_true = T_base_world @ d["T_world_cam"]
        T_base_cam = E @ T_base_cam_true
        table_z = float(d["table_top_z"] - d["T_world_base"][2, 3])
        gt = json.loads(str(d["boxes"]))
        gt_boxes = np.array([g["box"] for g in gt], float).reshape(-1, 4)
        gt_cls = np.array([g["cls"] for g in gt])
        id_to_center = {int(i): transform(T_base_world, c[None])[0]
                        for i, c in zip(d["obj_ids"], d["obj_center_world"])}

        r = model.predict(img[:, :, ::-1].copy(), imgsz=640, conf=args.conf, device="cpu", verbose=False)[0]
        pb = r.boxes.xyxy.numpy()
        pc = r.boxes.cls.numpy().astype(int)
        for i, j in match(pb, pc, gt_boxes, gt_cls):
            truth = id_to_center[gt[j]["id"]]
            for k, fn in METHODS.items():
                p = fn(d["depth"], d["K"], pb[i], T_base_cam, table_z_base=table_z)
                if p is not None:
                    errs[k]["xyz"].append(np.linalg.norm(p - truth))
                    errs[k]["xy"].append(np.linalg.norm(p[:2] - truth[:2]))
            p = METHODS[PIPELINE_METHOD](d["depth"], d["K"], pb[i], T_base_cam_true, table_z_base=table_z)
            if p is not None:
                errs["pipeline_true_extrinsics"]["xyz"].append(np.linalg.norm(p - truth))
                errs["pipeline_true_extrinsics"]["xy"].append(np.linalg.norm(p[:2] - truth[:2]))
            p = METHODS[PIPELINE_METHOD](d["depth"], d["K"], gt_boxes[j], T_base_cam, table_z_base=table_z)
            if p is not None:
                errs["pipeline_gt_boxes"]["xyz"].append(np.linalg.norm(p - truth))
                errs["pipeline_gt_boxes"]["xy"].append(np.linalg.norm(p[:2] - truth[:2]))
        if len(examples) < 2 and len(pb) >= 4:
            examples.append((img, pb, pc, r.boxes.conf.numpy(), d, T_base_cam, table_z))

    out = {k: {"xyz": stats(v["xyz"]), "xy": stats(v["xy"])} for k, v in errs.items() if v["xyz"]}
    out["notes"] = ("errors against the simulator's object bounding box centre, robot base frame; "
                    "detections with confidence >= %.2f matched to ground truth at IoU >= 0.5" % args.conf)
    (ROOT / "results/lifting_3d.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # figure 1: error by method
    keys = ["center_pixel", "box_median", "surface_median", "surface_symmetric"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.boxplot([np.array(errs[k]["xy"]) * 1000 for k in keys], showfliers=False)
    ax.set_xticks(range(1, len(keys) + 1), ["centre pixel", "box median", "surface median", "surface +\nsymmetric (used)"])
    ax.set_ylabel("horizontal (xy) error in base frame (mm)")
    ax.set_title("Where to put the grasp: error of four 3D lifting strategies")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ROOT / "results/lifting_error_by_method.png", dpi=130)

    # figure 2: example detections with their 3D positions in the robot base frame
    tiles = []
    for img, pb, pc, conf, d, T_base_cam, table_z in examples:
        vis = img.copy()
        for b, c, s in zip(pb, pc, conf):
            p = METHODS[PIPELINE_METHOD](d["depth"], d["K"], b, T_base_cam, table_z_base=table_z)
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 200, 40), 2)
            cv2.putText(vis, f"{CLASS_NAMES[c]} {s:.2f}", (x1, max(y1 - 16, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (20, 20, 220), 1, cv2.LINE_AA)
            if p is not None:
                cv2.putText(vis, f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m", (x1, max(y1 - 4, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 60, 20), 1, cv2.LINE_AA)
        tiles.append(vis)
    if tiles:
        cv2.imwrite(str(ROOT / "results/example_detections_3d.png"), cv2.hconcat(tiles))


if __name__ == "__main__":
    main()
