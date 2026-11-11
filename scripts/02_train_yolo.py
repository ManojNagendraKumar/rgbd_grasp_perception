"""Fine tune a COCO pretrained YOLOv8n on the simulated workcell dataset (CPU friendly settings)."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")               # transfer learning from COCO weights
    model.train(
        data=str(ROOT / "data/dataset.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device="cpu", workers=2, project=str(ROOT / "runs"), name="yolov8n_workcell", exist_ok=True,
        seed=0, deterministic=True, patience=50, plots=True,
        # augmentation: colour jitter for lighting robustness; no vertical flips (tables are not upside down)
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, fliplr=0.5, flipud=0.0, mosaic=1.0, degrees=0.0,
    )
    # the evaluation scripts read the weights from models/best.pt
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(ROOT / "runs/yolov8n_workcell/weights/best.pt", ROOT / "models/best.pt")


if __name__ == "__main__":
    main()
