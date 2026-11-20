"""Vertical correction (pitch + roll) from vanishing-direction estimation.

Assumes a rectilinear (non-fisheye) lens, a roughly level camera and landscape frames.
It corrects keystone and tilt of vertical lines. It does NOT fix yaw (corner / angled shots);
those are a documented limitation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

E_Y = np.array([0.0, 1.0, 0.0])


@dataclass
class Result:
    status: str                 # ok | skipped_no_lines | skipped_too_tilted
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    dev_before: float | None = None   # median |angle from vertical| of near-vertical lines, degrees
    dev_after: float | None = None
    n_lines: int = 0
    n_inliers: int = 0
    crop_scale: float = 1.0
    homography: np.ndarray | None = None
    crop: tuple[int, int, int, int] | None = None   # x0, y0, w, h in the full-res warped canvas


def camera_matrix(w: int, h: int, focal_mm: float, sensor_w_mm: float = 36.0) -> np.ndarray:
    f = focal_mm / sensor_w_mm * w
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def detect_segments(gray: np.ndarray) -> np.ndarray:
    """Return Nx4 array of line segments (x1, y1, x2, y2)."""
    try:
        lines = cv2.createLineSegmentDetector().detect(gray)[0]
        if lines is not None:
            return lines.reshape(-1, 4).astype(np.float64)
    except cv2.error:
        pass
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 720, 80,
                            minLineLength=max(gray.shape) // 25, maxLineGap=5)
    return np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(np.float64)


def _vertical_dev(segs: np.ndarray) -> np.ndarray:
    d = segs[:, 2:] - segs[:, :2]
    return np.degrees(np.arctan2(d[:, 0], d[:, 1]))  # 0 = vertical
    

def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + 90.0) % 180.0 - 90.0


def select_vertical(segs: np.ndarray, min_len: float, max_dev_deg: float = 20.0) -> np.ndarray:
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    dev = np.abs(_wrap(_vertical_dev(segs)))
    return segs[(length >= min_len) & (dev <= max_dev_deg)]


def _normals(segs: np.ndarray, K: np.ndarray):
    Ki = np.linalg.inv(K)
    p1 = np.c_[segs[:, 0], segs[:, 1], np.ones(len(segs))] @ Ki.T
    p2 = np.c_[segs[:, 2], segs[:, 3], np.ones(len(segs))] @ Ki.T
    n = np.cross(p1, p2)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return n, w


def _orient(d: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    return d if d @ E_Y >= 0 else -d


def estimate_vertical_direction(segs: np.ndarray, K: np.ndarray, iters: int = 500,
                                tol: float = 0.008, seed: int = 0):
    """Dominant 3D direction of the vertical lines (camera coordinates) + inlier mask."""
    n, w = _normals(segs, K)
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.choice(len(n), 2, replace=False, p=p)
        d = np.cross(n[i], n[j])
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        d = _orient(d)
        score = w[np.abs(n @ d) < tol].sum()
        if score > best_score:
            best, best_score = d, score
    if best is None:
        return None, np.zeros(len(n), bool)
    d = best
    for _ in range(3):  # weighted least-squares refinement on the inliers
        inl = np.abs(n @ d) < tol
        if inl.sum() < 2:
            break
        A = n[inl] * w[inl, None]
        d = _orient(np.linalg.svd(A, full_matrices=False)[2][-1])
    return d, np.abs(n @ d) < tol


def rotation_to_vertical(d: np.ndarray) -> np.ndarray:
    axis = np.cross(d, E_Y)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    angle = np.arctan2(s, d @ E_Y)
    return cv2.Rodrigues(axis / s * angle)[0]


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def inscribed_rect(quad: np.ndarray, aspect: float, w0: int, h0: int):
    """Largest axis-aligned rectangle with the given aspect inside the convex quad (grid search)."""
    quad32 = quad.astype(np.float32).reshape(-1, 1, 2)
    cx, cy = quad.mean(axis=0)
    span = np.ptp(quad, axis=0)

    def inside(x0, y0, rw, rh):
        for px, py in ((x0, y0), (x0 + rw, y0), (x0 + rw, y0 + rh), (x0, y0 + rh)):
            if cv2.pointPolygonTest(quad32, (float(px), float(py)), False) < 0:
                return False
        return True

    best = (0.0, None)
    for ox in np.linspace(-0.1, 0.1, 9):
        for oy in np.linspace(-0.1, 0.1, 9):
            px, py = cx + ox * span[0], cy + oy * span[1]
            lo, hi = 0.0, 1.0
            for _ in range(18):
                mid = (lo + hi) / 2
                rw, rh = mid * w0, mid * h0
                if inside(px - rw / 2, py - rh / 2, rw, rh):
                    lo = mid
                else:
                    hi = mid
            if lo > best[0]:
                best = (lo, (px, py))
    s, (px, py) = best
    rw, rh = int(s * w0) // 2 * 2, int(s * h0) // 2 * 2
    return s, (int(round(px - rw / 2)), int(round(py - rh / 2)), rw, rh)


def estimate(gray_small: np.ndarray, full_size: tuple[int, int], focal_mm: float,
             max_pitch: float = 12.0, max_roll: float = 6.0,
             sensor_w_mm: float = 36.0) -> Result:
    """Estimate the correction on a small grayscale copy; return a homography for full size."""
    hs, ws = gray_small.shape[:2]
    wf, hf = full_size
    K_small = camera_matrix(ws, hs, focal_mm, sensor_w_mm)
    segs = detect_segments(gray_small)
    vert = select_vertical(segs, min_len=0.03 * max(hs, ws)) if len(segs) else np.zeros((0, 4))
    if len(vert) < 8:
        return Result("skipped_no_lines", n_lines=len(vert))
    dev_before = float(np.median(np.abs(_wrap(_vertical_dev(vert)))))
    d, inl = estimate_vertical_direction(vert, K_small)
    if d is None or inl.sum() < 6:
        return Result("skipped_no_lines", n_lines=len(vert), n_inliers=int(inl.sum()),
                      dev_before=dev_before)
    R = rotation_to_vertical(d)
    rvec = cv2.Rodrigues(R)[0].ravel()
    pitch, roll = float(np.degrees(rvec[0])), float(np.degrees(rvec[2]))
    Hs = K_small @ R @ np.linalg.inv(K_small)
    p1 = _apply(Hs, vert[:, :2]); p2 = _apply(Hs, vert[:, 2:])
    dev_after = float(np.median(np.abs(_wrap(_vertical_dev(np.c_[p1, p2])))))
    base = Result("ok", pitch, roll, dev_before, dev_after, len(vert), int(inl.sum()))
    if abs(pitch) > max_pitch or abs(roll) > max_roll:
        base.status = "skipped_too_tilted"
        return base
    K = camera_matrix(wf, hf, focal_mm, sensor_w_mm)
    H = K @ R @ np.linalg.inv(K)
    corners = np.array([[0, 0], [wf, 0], [wf, hf], [0, hf]], float)
    s, (x0, y0, rw, rh) = inscribed_rect(_apply(H, corners), wf / hf, wf, hf)
    base.homography, base.crop_scale, base.crop = H, s, (x0, y0, rw, rh)
    return base


def warp(img: np.ndarray, res: Result) -> np.ndarray:
    """Apply the estimated correction and crop to the inscribed rectangle."""
    if res.homography is None or res.crop is None:
        return img
    x0, y0, rw, rh = res.crop
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
    return cv2.warpPerspective(img, T @ res.homography, (rw, rh), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
