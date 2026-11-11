"""Measure detector latency on CPU: PyTorch vs ONNX Runtime, batch 1 and batch 8.

Latency is measured end to end per image (preprocess + inference + NMS), after warm up, and reported as
mean, p50 and p90. Writes results/latency.json.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(model, images, imgsz, batch, runs=40, warmup=5):
    times = []
    for i in range(warmup + runs):
        batch_imgs = [images[(i * batch + k) % len(images)] for k in range(batch)]
        t0 = time.perf_counter()
        model.predict(batch_imgs, imgsz=imgsz, conf=0.25, device="cpu", verbose=False)
        dt = (time.perf_counter() - t0) * 1000 / batch
        if i >= warmup:
            times.append(dt)
    t = np.array(times)
    return {"ms_per_image_mean": float(t.mean()), "ms_per_image_p50": float(np.percentile(t, 50)),
            "ms_per_image_p90": float(np.percentile(t, 90)), "images_per_second": float(1000 / t.mean())}


def main():
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    imgsz = 512
    weights = ROOT / "models/best.pt"
    images = [cv2.imread(str(p))[:, :, ::-1].copy()
              for p in sorted((ROOT / "data/yolo/images/test").glob("*.png"))[:32]]

    out = {"hardware": "2 CPU cores, no GPU", "imgsz": imgsz}
    pt = YOLO(str(weights))
    out["pytorch_batch1"] = measure(pt, images, imgsz, 1)
    out["pytorch_batch8"] = measure(pt, images, imgsz, 8, runs=10)

    onnx_path = pt.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    ox = YOLO(onnx_path, task="detect")
    out["onnxruntime_batch1"] = measure(ox, images, imgsz, 1)
    out["onnx_speedup_batch1"] = out["pytorch_batch1"]["ms_per_image_mean"] / out["onnxruntime_batch1"][
        "ms_per_image_mean"]

    # does the exported model give the same detections? compare boxes on a few test images
    diffs = []
    for img in images[:10]:
        a = pt.predict(img, imgsz=imgsz, conf=0.5, device="cpu", verbose=False)[0].boxes.xyxy.numpy()
        b = ox.predict(img, imgsz=imgsz, conf=0.5, device="cpu", verbose=False)[0].boxes.xyxy.numpy()
        if len(a) == len(b) and len(a):
            diffs.append(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).max())
    out["onnx_vs_pytorch_max_box_diff_px"] = float(max(diffs)) if diffs else None
    (ROOT / "results/latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
