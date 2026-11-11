"""Generate a labelled RGBD dataset from the simulated workcell.

Output:
  data/yolo/images/{train,val,test}/*.png     RGB images
  data/yolo/labels/{train,val,test}/*.txt     YOLO boxes (class cx cy w h, normalised)
  data/test_rgbd/*.npz                        depth, ground truth masks, intrinsics, extrinsics, object poses
  data/dataset.yaml                           Ultralytics dataset file

Each split uses its own random seeds, so test scenes (layouts, lighting, camera poses) are never seen in training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.sim_scene import CLASS_NAMES, Workcell, add_depth_noise  # noqa: E402

MIN_VISIBLE_PX = 60  # boxes with fewer visible pixels are too occluded to label reliably


def boxes_from_mask(frame):
    out = []
    h, w = frame.seg.shape
    for o in frame.objects:
        m = frame.seg == o["id"]
        n = int(m.sum())
        if n < MIN_VISIBLE_PX:
            continue
        ys, xs = np.nonzero(m)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        out.append({"id": o["id"], "cls": CLASS_NAMES.index(o["cls"]), "box": [int(x1), int(y1), int(x2), int(y2)],
                    "visible_px": n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--test", type=int, default=150)
    args = ap.parse_args()

    data = ROOT / "data"
    splits = {"train": (args.train, 1000), "val": (args.val, 50000), "test": (args.test, 90000)}
    stats = {}
    t0 = time.time()
    for split, (n, seed0) in splits.items():
        (data / "yolo/images" / split).mkdir(parents=True, exist_ok=True)
        (data / "yolo/labels" / split).mkdir(parents=True, exist_ok=True)
        if split == "test":
            (data / "test_rgbd").mkdir(parents=True, exist_ok=True)
        wc = Workcell(seed=seed0)
        n_boxes = 0
        for i in range(n):
            wc.spawn_random_objects(3, 7)
            wc.add_distractors(2)
            wc.randomize_appearance()
            wc.randomize_camera()
            f = wc.render(lighting="random")
            depth = add_depth_noise(f.depth, wc.rng, f_px=f.K[0, 0])
            boxes = boxes_from_mask(f)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(data / "yolo/images" / split / f"{name}.png"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
            hh, ww = f.seg.shape
            with open(data / "yolo/labels" / split / f"{name}.txt", "w") as fh:
                for b in boxes:
                    x1, y1, x2, y2 = b["box"]
                    fh.write(f"{b['cls']} {(x1 + x2) / 2 / ww:.6f} {(y1 + y2) / 2 / hh:.6f} "
                             f"{(x2 - x1) / ww:.6f} {(y2 - y1) / hh:.6f}\n")
            n_boxes += len(boxes)
            if split == "test":
                np.savez_compressed(
                    data / "test_rgbd" / f"{name}.npz",
                    depth=depth.astype(np.float32), seg=f.seg, K=f.K, T_world_cam=f.T_world_cam,
                    T_world_base=wc.T_world_base, table_top_z=wc.table_top_z,
                    obj_ids=np.array([o["id"] for o in f.objects]),
                    obj_cls=np.array([CLASS_NAMES.index(o["cls"]) for o in f.objects]),
                    obj_center_world=np.array([o["aabb_center_world"] for o in f.objects]),
                    boxes=json.dumps(boxes))
        wc.close()
        stats[split] = {"images": n, "boxes": n_boxes}
        print(split, stats[split], f"{time.time() - t0:.0f}s", flush=True)

    (data / "dataset.yaml").write_text(
        f"path: {data / 'yolo'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES)))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/dataset_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
