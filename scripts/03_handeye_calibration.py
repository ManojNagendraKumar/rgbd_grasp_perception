"""Eye to hand calibration of the cell camera, and how its accuracy depends on the number of robot poses.

Writes results/handeye.json and results/handeye_error_vs_poses.png.
The 15 pose Tsai result is saved as the calibration used by the 3D lifting evaluation.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.geometry import invert  # noqa: E402
from perception.handeye import METHODS, calibration_error, collect_measurements, solve_eye_to_hand  # noqa: E402
from perception.sim_scene import CameraConfig, Workcell  # noqa: E402
from perception.sim_scene import view_to_T_world_cam  # noqa: E402
import pybullet as p  # noqa: E402


def main():
    wc = Workcell(seed=7)
    cam = CameraConfig()
    view = p.computeViewMatrix(cam.eye, cam.target, cam.up)
    T_base_cam_true = invert(wc.T_world_base) @ view_to_T_world_cam(view)
    K = cam.K

    pose_counts = [4, 6, 8, 10, 15, 20, 30]
    trials = 20
    rng = np.random.default_rng(0)
    table = {m: {n: [] for n in pose_counts} for m in METHODS}
    for trial in range(trials):
        meas = collect_measurements(wc, T_base_cam_true, K, (cam.width, cam.height), max(pose_counts), rng)
        for n in pose_counts:
            for m in METHODS:
                err = calibration_error(solve_eye_to_hand(meas[:n], m), T_base_cam_true)
                table[m][n].append(err)

    summary = {m: {n: {"rot_deg_median": float(np.median([e["rot_deg"] for e in table[m][n]])),
                       "trans_mm_median": float(np.median([e["trans_mm"] for e in table[m][n]]))}
                   for n in pose_counts} for m in METHODS}

    # the calibration actually used downstream: 15 poses, Tsai, one fixed run
    meas = collect_measurements(wc, T_base_cam_true, K, (cam.width, cam.height), 15, np.random.default_rng(123))
    T_est = solve_eye_to_hand(meas, "Tsai")
    used = calibration_error(T_est, T_base_cam_true)
    np.savez(ROOT / "results/handeye_calibration.npz", T_base_cam_est=T_est, T_base_cam_true=T_base_cam_true)

    out = {"setup": {"pixel_noise_px": 0.3, "board": "7 x 6 corners, 25 mm squares", "trials_per_point": trials},
           "median_error_by_method_and_pose_count": summary,
           "calibration_used_downstream": {"method": "Tsai", "poses": 15, **used}}
    (ROOT / "results/handeye.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in METHODS:
        ax[0].plot(pose_counts, [summary[m][n]["trans_mm_median"] for n in pose_counts], marker="o", label=m)
        ax[1].plot(pose_counts, [summary[m][n]["rot_deg_median"] for n in pose_counts], marker="o", label=m)
    ax[0].set(xlabel="robot poses", ylabel="translation error (mm)", title="Camera position error")
    ax[1].set(xlabel="robot poses", ylabel="rotation error (deg)", title="Camera rotation error")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_yscale("log")
    ax[1].legend(fontsize=8)
    fig.suptitle("Eye to hand calibration accuracy vs number of poses (median of 20 runs, 0.3 px corner noise)")
    fig.tight_layout()
    fig.savefig(ROOT / "results/handeye_error_vs_poses.png", dpi=130)
    print(json.dumps(out["calibration_used_downstream"], indent=2))
    for m in METHODS:
        print(m, {n: (round(summary[m][n]["trans_mm_median"], 2), round(summary[m][n]["rot_deg_median"], 3))
                  for n in pose_counts})


if __name__ == "__main__":
    main()
