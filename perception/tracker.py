"""Multi object tracking: a constant velocity Kalman filter, Hungarian association and ByteTrack.

ByteTrack (Zhang et al., ECCV 2022) is implemented from the paper's algorithm, not imported, so every step
can be explained. Setting use_low_score_pass=False gives the SORT style baseline that keeps only high
confidence detections; comparing the two isolates the effect of ByteTrack's second association.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- geometry helpers
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ---------------------------------------------------------------- Kalman filter
class KalmanFilterXYAH:
    """State: box centre (cx, cy), aspect ratio a = w/h, height h, and their velocities.
    Noise scales with box height, so large and small boxes are treated consistently."""

    def __init__(self, std_pos=1 / 20, std_vel=1 / 160):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)        # x_k+1 = x_k + v_k (dt = 1 frame)
        self.H = np.eye(4, 8)
        self.wp, self.wv = std_pos, std_vel

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.wp * h, 2 * self.wp * h, 1e-2, 2 * self.wp * h,
               10 * self.wv * h, 10 * self.wv * h, 1e-5, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = [self.wp * h, self.wp * h, 1e-2, self.wp * h, self.wv * h, self.wv * h, 1e-5, self.wv * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(q))
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov


# ---------------------------------------------------------------- tracks
TRACKED, LOST, REMOVED = 0, 1, 2


class Track:
    _next_id = 1

    def __init__(self, box, score, cls, kf, frame_id, activate):
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.score, self.cls = score, cls
        self.state = TRACKED
        self.activated = activate
        self.id = self._take_id() if activate else 0
        self.frame_id = self.start_frame = frame_id

    @classmethod
    def _take_id(cls):
        i = cls._next_id
        cls._next_id += 1
        return i

    @classmethod
    def reset_ids(cls):
        cls._next_id = 1

    def predict(self):
        if self.state != TRACKED:
            self.mean[7] = 0.0  # do not let a lost track's height keep growing
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    def update(self, box, score, cls, frame_id):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(box))
        self.score, self.cls, self.frame_id = score, cls, frame_id
        self.state = TRACKED
        if not self.activated:
            self.activated = True
            self.id = self._take_id()

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)


def associate(tracks, boxes, max_cost):
    """Hungarian assignment on cost = 1 - IoU. Pairs with cost above max_cost are rejected."""
    if len(tracks) == 0 or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    cost = 1.0 - iou_matrix(np.array([t.box for t in tracks]), boxes)
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
    mt, md = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]


class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6, match_cost=0.8,
                 low_match_cost=0.5, track_buffer=30, use_low_score_pass=True):
        self.high, self.low, self.new_thr = high_thresh, low_thresh, new_track_thresh
        self.match_cost, self.low_match_cost = match_cost, low_match_cost
        self.buffer, self.use_low = track_buffer, use_low_score_pass
        self.kf = KalmanFilterXYAH()
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.frame_id = 0
        Track.reset_ids()

    def update(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray):
        self.frame_id += 1
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores, classes = np.asarray(scores, dtype=float), np.asarray(classes)
        hi = scores >= self.high
        lo = (scores >= self.low) & ~hi
        hb, hs, hc = boxes[hi], scores[hi], classes[hi]
        lb, ls, lc = boxes[lo], scores[lo], classes[lo]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.predict()

        # 1) high score detections against every confirmed or lost track
        matches, um_tracks, um_dets = associate(pool, hb, self.match_cost)
        for r, c in matches:
            pool[r].update(hb[c], hs[c], hc[c], self.frame_id)

        # 2) ByteTrack's key step: tracks still unmatched against the LOW score detections.
        #    An occluded object usually still gets a box, just with low confidence.
        remaining = [pool[i] for i in um_tracks if pool[i].state == TRACKED]
        if self.use_low and len(lb):
            m2, um2, _ = associate(remaining, lb, self.low_match_cost)
            for r, c in m2:
                remaining[r].update(lb[c], ls[c], lc[c], self.frame_id)
            remaining = [remaining[i] for i in um2]
        for t in remaining:
            t.state = LOST

        # 3) tentative tracks from the previous frame get one chance to be confirmed
        left_b, left_s, left_c = hb[um_dets], hs[um_dets], hc[um_dets]
        m3, um3, um_d3 = associate(unconfirmed, left_b, 0.7)
        for r, c in m3:
            unconfirmed[r].update(left_b[c], left_s[c], left_c[c], self.frame_id)
        for i in um3:
            unconfirmed[i].state = REMOVED

        # 4) new tentative tracks from confident unmatched detections
        new = [Track(left_b[j], left_s[j], left_c[j], self.kf, self.frame_id, activate=(self.frame_id == 1))
               for j in um_d3 if left_s[j] >= self.new_thr]

        # 5) bookkeeping: drop tracks lost for longer than the buffer
        everyone = pool + unconfirmed + new
        for t in everyone:
            if t.state == LOST and self.frame_id - t.frame_id > self.buffer:
                t.state = REMOVED
        self.tracked = [t for t in everyone if t.state == TRACKED]
        self.lost = [t for t in everyone if t.state == LOST]
        return [(t.id, t.box.copy(), t.score, t.cls) for t in self.tracked if t.activated]
