"""Compare ByteTrack with a SORT style baseline on the rendered sequences, using the trained detector.

Both trackers share the same Kalman filter and Hungarian association. The only difference is ByteTrack's
second association pass over low confidence detections, so the comparison isolates that idea.
Metrics are standard CLEAR MOT and IDF1 from the motmetrics library (IoU >= 0.5 to count a match).

Writes results/tracking.json and results/tracking_example.png.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.tracker import ByteTracker, iou_matrix  # noqa: E402

TRACKERS = {
    "sort_style_baseline": dict(use_low_score_pass=False),
    "bytetrack": dict(use_low_score_pass=True),
}


def detect_sequence(model, seq_dir, imgsz):
    dets = []
    for fpath in sorted(seq_dir.glob("frame_*.png")):
        img = cv2.imread(str(fpath))[:, :, ::-1].copy()
        r = model.predict(img, imgsz=imgsz, conf=0.05, device="cpu", verbose=False)[0]
        dets.append((r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int)))
    return dets


def run_tracker(dets, cfg):
    trk = ByteTracker(**cfg)
    return [trk.update(b, s, c) for b, s, c in dets]


def evaluate(gt_frames, hyp_frames):
    acc = mm.MOTAccumulator(auto_id=True)
    for gt, hyp in zip(gt_frames, hyp_frames):
        g_ids = [g["id"] for g in gt]
        g_boxes = np.array([g["box"] for g in gt], float).reshape(-1, 4)
        h_ids = [h[0] for h in hyp]
        h_boxes = np.array([h[1] for h in hyp], float).reshape(-1, 4)
        dist = 1.0 - iou_matrix(g_boxes, h_boxes)
        dist[dist > 0.5] = np.nan          # IoU below 0.5 cannot be a match (same rule as motmetrics)
        acc.update(g_ids, h_ids, dist)
    return acc


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()
    model = YOLO(args.weights)

    seqs = sorted((ROOT / "data/tracking").iterdir())
    accs = {k: {} for k in TRACKERS}
    example = None
    for seq in seqs:
        gt = json.loads((seq / "gt.json").read_text())["frames"]
        dets = detect_sequence(model, seq, args.imgsz)
        for name, cfg in TRACKERS.items():
            hyp = run_tracker(dets, cfg)
            accs[name][seq.name] = evaluate(gt, hyp)
            if name == "bytetrack" and seq.name == "occluder_0":
                example = (seq, hyp)
        print("evaluated", seq.name, flush=True)

    mh = mm.metrics.create()
    metrics = ["num_switches", "num_fragmentations", "mota", "idf1", "num_misses", "num_false_positives",
               "num_objects"]
    out = {}
    for name in TRACKERS:
        per_cond = {}
        for cond in ("crossing", "occluder", "fast"):
            names = [s for s in accs[name] if s.startswith(cond)]
            summ = mh.compute_many([accs[name][s] for s in names], metrics=metrics, names=names,
                                   generate_overall=True)
            per_cond[cond] = {m: float(summ.loc["OVERALL", m]) for m in metrics}
        summ = mh.compute_many(list(accs[name].values()), metrics=metrics, names=list(accs[name]),
                               generate_overall=True)
        per_cond["all"] = {m: float(summ.loc["OVERALL", m]) for m in metrics}
        out[name] = per_cond
    b, s = out["bytetrack"]["all"]["num_switches"], out["sort_style_baseline"]["all"]["num_switches"]
    out["id_switch_reduction_percent"] = float(100 * (s - b) / s) if s > 0 else None
    out["setup"] = {"sequences": len(seqs), "frames_per_sequence": 80, "detector_conf_floor": 0.05,
                    "high_thresh": 0.5, "low_thresh": 0.1, "match": "IoU >= 0.5 (motmetrics)"}
    (ROOT / "results/tracking.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # example strip: four frames of the occluder sequence with ByteTrack identities
    if example:
        seq, hyp = example
        tiles = []
        for t in (20, 35, 45, 60):
            im = cv2.imread(str(seq / f"frame_{t:04d}.png"))
            for tid, box, score, cls in hyp[t]:
                x1, y1, x2, y2 = box.astype(int)
                col = tuple(int(c) for c in np.random.default_rng(tid).integers(60, 255, 3))
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
                cv2.putText(im, f"id {tid}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(im, f"frame {t}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(cv2.resize(im, (480, 360)))
        cv2.imwrite(str(ROOT / "results/tracking_example.png"),
                    cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])]))


if __name__ == "__main__":
    main()
