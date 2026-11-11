"""Compare a simulated frame with a real one, and run the simulation trained detector on the real frame.

The real frame is the public example of GraspNet 1Billion (Fang et al., CVPR 2020): a RealSense camera mounted on
a robot arm, looking at everyday objects on a table. It is downloaded from the graspnet-baseline GitHub repository,
not stored here. The simulated frame is test image 3 of the dataset, rendered again from the same seed.

None of the real objects belong to the detector's five classes, so this is not a score. It shows how a detector
trained only on simulation behaves on real pixels.

Output: results/sim_vs_real.png, results/sim_detector_on_real_frame.png, results/sim_to_real_check.json
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.sim_scene import Workcell, add_depth_noise  # noqa: E402

URL = "https://raw.githubusercontent.com/graspnet/graspnet-baseline/main/doc/example_data/"
WEIGHTS = ROOT / "models/best.pt"
TEST_SEED, TEST_INDEX = 90000, 3  # same seed and order as the test split in 01_generate_dataset.py
CONF, IMGSZ = 0.5, 512            # same settings as 04_evaluate_detection_3d.py


def simulated_test_frame():
    wc = Workcell(seed=TEST_SEED)
    for _ in range(TEST_INDEX + 1):
        wc.spawn_random_objects(3, 7)
        wc.add_distractors(2)
        wc.randomize_appearance()
        wc.randomize_camera()
        f = wc.render(lighting="random")
        depth = add_depth_noise(f.depth, wc.rng, f_px=f.K[0, 0])
    wc.close()
    return f.rgb, depth


def real_frame():
    dst = ROOT / "data/real_example"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("color.png", "depth.png"):
        if not (dst / name).exists():
            urllib.request.urlretrieve(URL + name, dst / name)
    rgb = np.array(Image.open(dst / "color.png").convert("RGB"))
    depth = np.array(Image.open(dst / "depth.png")).astype(np.float64) / 1000.0   # millimetres in the PNG
    return rgb, depth


def detect(model, rgb):
    r = model.predict(rgb[:, :, ::-1].copy(), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    return [{"class": r.names[int(c)], "score": round(float(s), 3), "box": [round(float(v), 1) for v in b]}
            for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ultralytics import YOLO

    sim_rgb, sim_depth = simulated_test_frame()
    real_rgb, real_depth = real_frame()
    real_depth[real_depth == 0] = np.nan
    missing = float(np.isnan(real_depth).mean())

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for a in ax.flat:
        a.axis("off")
    ax[0, 0].imshow(sim_rgb)
    ax[0, 0].set_title("This project: PyBullet render (colour)", fontsize=11)
    ax[0, 1].imshow(sim_depth, cmap="turbo")
    ax[0, 1].set_title("PyBullet depth (stereo noise model added by the code)", fontsize=11)
    ax[1, 0].imshow(real_rgb)
    ax[1, 0].set_title("Real: GraspNet frame, RealSense camera on a robot arm", fontsize=11)
    ax[1, 1].imshow(real_depth, cmap="turbo", vmin=np.nanpercentile(real_depth, 1),
                    vmax=np.nanpercentile(real_depth, 99))
    ax[1, 1].set_title(f"Real depth (white = no depth, {100 * missing:.0f}% of pixels)", fontsize=11)
    fig.savefig(ROOT / "results/sim_vs_real.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    model = YOLO(str(WEIGHTS))
    sim_det, real_det = detect(model, sim_rgb), detect(model, real_rgb)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[4 / 3, 16 / 9], wspace=0.05))
    for a, img, dets, title in ((ax[0], sim_rgb, sim_det, "Simulated test frame"),
                                (ax[1], real_rgb, real_det, "Real frame with objects it never saw")):
        a.imshow(img)
        a.axis("off")
        a.set_title(f"{title}: {len(dets)} detections", fontsize=11)
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=2))
            a.text(x1 + 3, y2 - 6, f"{d['class']} {d['score']:.2f}", color="black", fontsize=8,
                   bbox=dict(fc="yellow", ec="none", pad=1))
    fig.savefig(ROOT / "results/sim_detector_on_real_frame.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    out = {"confidence_threshold": CONF, "imgsz": IMGSZ, "real_missing_depth_fraction": missing,
           "simulated_frame_detections": sim_det, "real_frame_detections": real_det}
    (ROOT / "results/sim_to_real_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
