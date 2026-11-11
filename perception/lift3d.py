"""Turn a 2D detection plus a depth image into a 3D grasp target in the robot base frame.

Strategies, compared in scripts/04_evaluate_detection_3d.py:
  center_pixel       depth of the single pixel at the box centre (naive)
  box_median         median depth inside the shrunken box, back projected along the box centre ray
  surface_median     back project the box to the base frame, remove the table plane and the background,
                     take the median of the remaining surface points
  surface_symmetric  same surface points, then complete the hidden back half: the camera only sees the
                     front and top of an object, so the visible median is biased toward the camera.
                     Assume the object is about as deep as it is wide (seen from the camera) and that it
                     rests on the table, so its centre height is half its top height. (used in the pipeline)
"""
from __future__ import annotations

import numpy as np

from .geometry import backproject, transform


def _shrink(box, frac, shape):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(round(x1 + frac * w)), int(round(x2 - frac * w))
    y1, y2 = int(round(y1 + frac * h)), int(round(y2 - frac * h))
    H, W = shape
    return max(x1, 0), max(y1, 0), min(max(x2, x1 + 1), W), min(max(y2, y1 + 1), H)


def lift_center_pixel(depth, K, box, T_base_cam, **_):
    u = int(round((box[0] + box[2]) / 2))
    v = int(round((box[1] + box[3]) / 2))
    z = depth[min(max(v, 0), depth.shape[0] - 1), min(max(u, 0), depth.shape[1] - 1)]
    if z <= 0:
        return None
    return transform(T_base_cam, backproject(K, np.array([u], float), np.array([v], float), np.array([z])))[0]


def lift_box_median(depth, K, box, T_base_cam, **_):
    x1, y1, x2, y2 = _shrink(box, 0.2, depth.shape)
    patch = depth[y1:y2, x1:x2]
    z = patch[patch > 0]
    if z.size == 0:
        return None
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return transform(T_base_cam, backproject(K, np.array([u]), np.array([v]), np.array([np.median(z)])))[0]


def object_surface_points(depth, K, box, T_base_cam, table_z_base=0.0, table_margin=0.008, max_obj_size=0.15):
    """All depth pixels in the box, in the base frame, minus the table plane and anything far behind."""
    x1, y1, x2, y2 = _shrink(box, 0.05, depth.shape)
    patch = depth[y1:y2, x1:x2]
    vv, uu = np.nonzero(patch > 0)
    if vv.size == 0:
        return None
    z = patch[vv, uu].astype(float)
    pts = transform(T_base_cam, backproject(K, (uu + x1).astype(float), (vv + y1).astype(float), z))
    above = pts[:, 2] > table_z_base + table_margin          # remove the support plane
    if above.sum() < 5:
        return None
    pts, z = pts[above], z[above]
    near = z <= np.percentile(z, 10) + max_obj_size           # drop background objects behind the target
    return pts[near]


def lift_surface_median(depth, K, box, T_base_cam, table_z_base=0.0, **_):
    pts = object_surface_points(depth, K, box, T_base_cam, table_z_base)
    return None if pts is None else np.median(pts, axis=0)


def lift_surface_symmetric(depth, K, box, T_base_cam, table_z_base=0.0, **_):
    pts = object_surface_points(depth, K, box, T_base_cam, table_z_base)
    if pts is None:
        return None
    cam_xy = T_base_cam[:2, 3]
    view = np.median(pts[:, :2], axis=0) - cam_xy             # horizontal viewing direction
    view /= np.linalg.norm(view) + 1e-9
    side = np.array([-view[1], view[0]])
    s, q = pts[:, :2] @ view, pts[:, :2] @ side
    s_front, s_back = np.percentile(s, 2), np.percentile(s, 98)
    q_lo, q_hi = np.percentile(q, 2), np.percentile(q, 98)
    width = q_hi - q_lo
    s_center = s_front + max(width, s_back - s_front) / 2.0   # assume depth about equal to width
    q_center = (q_lo + q_hi) / 2.0
    top = np.percentile(pts[:, 2], 98)
    return np.r_[s_center * view + q_center * side, (top + table_z_base) / 2.0]


METHODS = {"center_pixel": lift_center_pixel, "box_median": lift_box_median,
           "surface_median": lift_surface_median, "surface_symmetric": lift_surface_symmetric}
PIPELINE_METHOD = "surface_symmetric"
