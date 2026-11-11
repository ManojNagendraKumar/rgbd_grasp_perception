"""Eye to hand calibration: find the fixed transform between the robot base and a camera mounted in the cell.

A calibration board is attached to the gripper. For each robot pose we know the gripper pose from forward
kinematics and measure the board pose in the camera with solvePnP. The unknown camera pose X satisfies
AX = XB; OpenCV solves it with several classic methods (Tsai, Park, Horaud, Andreff, Daniilidis).

In simulation the board corners are obtained by projecting the board with the true camera and adding pixel
noise, which mimics a corner detector. Everything after that (PnP, AX = XB, error analysis) is the real method.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import invert, project, rot_angle_deg

METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def board_points(cols=7, rows=6, square=0.025):
    g = np.array([[c * square, r * square, 0.0] for r in range(rows) for c in range(cols)])
    return g - g.mean(axis=0)


def random_rotation_near(R0, max_deg, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.radians(rng.uniform(0, max_deg))
    R, _ = cv2.Rodrigues(axis * ang)
    return R0 @ R


def collect_measurements(wc, T_base_cam_true, K, image_size, n_poses, rng, pixel_noise=0.3,
                         T_ee_board=None, max_tries=2000):
    """Move the simulated Panda to varied poses; return FK poses and PnP board poses."""
    w, h = image_size
    obj = board_points()
    if T_ee_board is None:
        T_ee_board = np.eye(4)
        T_ee_board[:3, 3] = [0.0, 0.0, 0.03]   # board 3 cm in front of the grasp frame
    T_cam_base = invert(T_base_cam_true)
    cam_pos_base = T_base_cam_true[:3, 3]
    out = []
    tries = 0
    while len(out) < n_poses and tries < max_tries:
        tries += 1
        pos = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.25, 0.25), rng.uniform(0.12, 0.40)])
        # nominal: board normal (+z of board) points toward the camera, then a random tilt up to 35 degrees
        z = cam_pos_base - pos
        z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_nominal = random_rotation_near(np.c_[x, y, z], 35, rng)
        quat = _mat_to_quat(R_nominal)
        T_base_ee = wc.move_ee_to(pos, quat)               # forward kinematics of the pose actually reached
        T_cam_board = T_cam_base @ T_base_ee @ T_ee_board
        pts_cam = (T_cam_board[:3, :3] @ obj.T).T + T_cam_board[:3, 3]
        if np.any(pts_cam[:, 2] < 0.1):
            continue
        normal_cam = T_cam_board[:3, 2]
        if normal_cam @ (-T_cam_board[:3, 3] / np.linalg.norm(T_cam_board[:3, 3])) < 0.3:
            continue                                        # board seen too edge on, a detector would fail
        uv = project(K, pts_cam) + rng.normal(scale=pixel_noise, size=(len(obj), 2))
        if np.any(uv < 5) or np.any(uv[:, 0] > w - 5) or np.any(uv[:, 1] > h - 5):
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, uv.astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R_meas, _ = cv2.Rodrigues(rvec)
        T_meas = np.eye(4)
        T_meas[:3, :3], T_meas[:3, 3] = R_meas, tvec.ravel()
        out.append({"T_base_ee": T_base_ee, "T_cam_board": T_meas})
    return out


def solve_eye_to_hand(measurements, method="Tsai"):
    """Returns T_base_cam. For eye to hand, OpenCV expects base->gripper instead of gripper->base."""
    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for m in measurements:
        T_ee_base = invert(m["T_base_ee"])
        R_b2g.append(T_ee_base[:3, :3])
        t_b2g.append(T_ee_base[:3, 3])
        R_t2c.append(m["T_cam_board"][:3, :3])
        t_t2c.append(m["T_cam_board"][:3, 3])
    R, t = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=METHODS[method])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.ravel()
    return T


def calibration_error(T_est, T_true):
    dR = T_est[:3, :3] @ T_true[:3, :3].T
    return {"rot_deg": rot_angle_deg(dR), "trans_mm": float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000)}


def _mat_to_quat(R):
    """Rotation matrix to (x, y, z, w) quaternion, the order PyBullet uses."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [x, y, z, w]
