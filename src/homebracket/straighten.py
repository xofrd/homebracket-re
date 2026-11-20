"""Orientation correction (pitch, roll and yaw) from line vanishing directions.

Assumes a rectilinear (non-fisheye) lens, landscape frames and a roughly square-on view.
Vertical lines give pitch and roll; the front-facing horizontal family gives the lateral turn
(yaw). Votes are balanced between regions (left/right for verticals, top/bottom for horizontals)
so that one crowded side cannot leave the other side uncalibrated. An optional region of
interest (ROI) restricts the lines used, for example to the house itself on a facade.
Angled/corner shots beyond the limits are reported, not forced.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

E_Y = np.array([0.0, 1.0, 0.0])


@dataclass
class Result:
    status: str                 # ok | ok_horizontals_only | skipped_no_lines | skipped_too_tilted
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    dev_before: float | None = None   # median |angle from vertical| of near-vertical lines, degrees
    dev_after: float | None = None
    n_lines: int = 0
    n_inliers: int = 0
    crop_scale: float = 1.0
    yaw_deg: float = 0.0              # lateral turn corrected using the horizontal lines
    n_h_lines: int = 0
    dev_h_before: float | None = None
    dev_h_after: float | None = None
    yaw_note: str = ""
    # residual per region after correction (median |deviation|, degrees; None = too few lines)
    dev_v_left: float | None = None
    dev_v_right: float | None = None
    dev_h_top: float | None = None
    dev_h_bottom: float | None = None
    homography: np.ndarray | None = None
    crop: tuple[int, int, int, int] | None = None   # x0, y0, w, h in the full-res warped canvas

    @property
    def worst_deg(self) -> float | None:
        vals = [v for v in (self.dev_v_left, self.dev_v_right, self.dev_h_top, self.dev_h_bottom)
                if v is not None]
        return max(vals) if vals else None


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


def _lengths(segs: np.ndarray) -> np.ndarray:
    return np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])


def select_vertical(segs: np.ndarray, min_len: float, max_dev_deg: float = 20.0) -> np.ndarray:
    dev = np.abs(_wrap(_vertical_dev(segs)))
    return segs[(_lengths(segs) >= min_len) & (dev <= max_dev_deg)]


def in_roi(segs: np.ndarray, size: tuple[int, int], roi) -> np.ndarray:
    """Keep segments whose midpoint lies inside roi = (x0, y0, x1, y1) as fractions of the image."""
    if roi is None or len(segs) == 0:
        return segs
    w, h = size
    mx = (segs[:, 0] + segs[:, 2]) / 2 / w
    my = (segs[:, 1] + segs[:, 3]) / 2 / h
    keep = (mx >= roi[0]) & (mx <= roi[2]) & (my >= roi[1]) & (my <= roi[3])
    return segs[keep]


def region_weights(w: np.ndarray, region: np.ndarray, cap: float = 3.0, min_lines: int = 3):
    """Rebalance weights so the two regions count about equally (amplification capped)."""
    w = np.asarray(w, float).copy()
    if len(w) == 0 or min((region == 0).sum(), (region == 1).sum()) < min_lines:
        return w
    tot = [w[region == k].sum() for k in (0, 1)]
    s = sum(tot)
    for k in (0, 1):
        w[region == k] *= np.clip(0.5 * s / tot[k], 1.0 / cap, cap)
    return w


def _normals(segs: np.ndarray, K: np.ndarray):
    Ki = np.linalg.inv(K)
    p1 = np.c_[segs[:, 0], segs[:, 1], np.ones(len(segs))] @ Ki.T
    p2 = np.c_[segs[:, 2], segs[:, 3], np.ones(len(segs))] @ Ki.T
    n = np.cross(p1, p2)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n, _lengths(segs)


def _orient(d: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    return d if d @ E_Y >= 0 else -d


def estimate_vertical_direction(segs: np.ndarray, K: np.ndarray, weights: np.ndarray | None = None,
                                iters: int = 500, tol: float = 0.008, seed: int = 0):
    """Dominant 3D direction of the vertical lines (camera coordinates) + inlier mask."""
    n, wlen = _normals(segs, K)
    w = wlen if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.choice(len(n), 2, replace=False, p=p)
        d = np.cross(n[i], n[j])
        if np.linalg.norm(d) < 1e-6:
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


def _hist_mode(angles: np.ndarray, weights: np.ndarray, lo: float, hi: float,
               bw: float = 0.4, step: float = 0.05):
    """Weighted kernel-density mode of angles (degrees) inside [lo, hi]."""
    grid = np.arange(lo, hi + step, step)
    dens = np.zeros_like(grid)
    for a, w in zip(angles, weights):
        dens += w * np.exp(-0.5 * ((grid - a) / bw) ** 2)
    return float(grid[int(np.argmax(dens))])


def estimate_yaw(segs: np.ndarray, K: np.ndarray, R_v: np.ndarray, min_len: float,
                 max_yaw: float, split_y: float):
    """After the verticals are fixed, find the dominant front-facing horizontal direction.

    In the vertically-corrected frame a horizontal 3D direction is (cos a, 0, sin a) and every
    line with plane normal n satisfies n . d = 0, so each line votes for one angle
    a = atan2(-n_x, n_z). Votes are balanced between the top and bottom halves.
    Returns (a_deg or None, candidate segs, support mask, note).
    """
    dev = np.abs(_wrap(_vertical_dev(segs)))          # 90 = horizontal in the image
    cand = segs[(_lengths(segs) >= min_len) & (dev >= 55)]
    if len(cand) < 6:
        return None, cand, np.zeros(len(cand), bool), "too few horizontal lines"
    n, w = _normals(cand, K)
    a = np.degrees(np.arctan2(-(n @ R_v.T)[:, 0], (n @ R_v.T)[:, 2]))
    a = (a + 90.0) % 180.0 - 90.0
    region = ((cand[:, 1] + cand[:, 3]) / 2 >= split_y).astype(int)  # 0 top, 1 bottom
    near = np.abs(a) <= max_yaw + 3.0
    if near.sum() < 6:
        return None, cand, near, "no frontal horizontal family (angled view?)"
    wb = np.zeros_like(w)
    wb[near] = region_weights(w[near], region[near])
    peak = _hist_mode(a[near], wb[near], -max_yaw - 3.0, max_yaw + 3.0, bw=0.5)
    sup = near & (np.abs(a - peak) <= 0.8)
    if sup.sum() < 6:
        return None, cand, sup, "horizontal family too weak"
    ws = np.zeros_like(w)
    ws[sup] = region_weights(w[sup], region[sup])
    peak = float(np.average(a[sup], weights=ws[sup]))
    if abs(peak) > max_yaw:
        return None, cand, sup, f"yaw {peak:.1f} deg exceeds limit {max_yaw:.0f} (angled view): not corrected"
    return peak, cand, sup, ""


def _h_dev(segs: np.ndarray) -> float:
    d = segs[:, 2:] - segs[:, :2]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.median(np.abs((ang + 90.0) % 180.0 - 90.0)))


def _v_dev(segs: np.ndarray) -> float:
    return float(np.median(np.abs(_wrap(_vertical_dev(segs)))))


def _median_or_none(vals, minimum=2):
    return float(np.median(vals)) if len(vals) >= minimum else None


def _roll_from_horizontals(segs, min_len, split_y, max_roll):
    """Fallback when there are no usable verticals: level the dominant horizontal family."""
    dev = np.abs(_wrap(_vertical_dev(segs)))
    cand = segs[(_lengths(segs) >= min_len) & (dev >= 70)]
    if len(cand) < 5:
        return None, cand
    d = cand[:, 2:] - cand[:, :2]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    region = ((cand[:, 1] + cand[:, 3]) / 2 >= split_y).astype(int)
    w = region_weights(_lengths(cand), region)
    peak = _hist_mode(ang, w, -max_roll, max_roll, bw=0.3)
    sup = np.abs(ang - peak) <= 0.6
    if sup.sum() < 4:
        return None, cand
    return float(np.average(ang[sup], weights=w[sup])), cand[sup]


def estimate(gray_small: np.ndarray, full_size: tuple[int, int], focal_mm: float,
             max_pitch: float = 12.0, max_roll: float = 6.0, max_yaw: float = 6.0,
             use_horizontals: bool = True, roi=None, sensor_w_mm: float = 36.0) -> Result:
    """Estimate the correction on a small grayscale copy; return a homography for full size.

    roi = (x0, y0, x1, y1) as image fractions: only lines whose midpoint is inside are used,
    and the left/right and top/bottom balance is computed relative to the ROI center.
    """
    hs, ws = gray_small.shape[:2]
    wf, hf = full_size
    K_small = camera_matrix(ws, hs, focal_mm, sensor_w_mm)
    segs = in_roi(detect_segments(gray_small), (ws, hs), roi)
    split_x = (roi[0] + roi[2]) / 2 * ws if roi else ws / 2
    split_y = (roi[1] + roi[3]) / 2 * hs if roi else hs / 2

    vert = np.zeros((0, 4))
    for frac in (0.03, 0.015):  # retry with shorter lines when the scene has few verticals
        min_len = frac * max(hs, ws)
        vert = select_vertical(segs, min_len) if len(segs) else vert
        if len(vert) >= 8:
            break

    R_v = None
    inl = np.zeros(len(vert), bool)
    dev_before = None
    if len(vert) >= 8:
        dev_before = _v_dev(vert)
        region_v = ((vert[:, 0] + vert[:, 2]) / 2 >= split_x).astype(int)  # 0 left, 1 right
        wv = region_weights(_lengths(vert), region_v)
        d, inl = estimate_vertical_direction(vert, K_small, weights=wv)
        if d is not None and inl.sum() >= 6:
            cand_R = rotation_to_vertical(d)
            Hc = K_small @ cand_R @ np.linalg.inv(K_small)
            vc = np.c_[_apply(Hc, vert[:, :2]), _apply(Hc, vert[:, 2:])]
            # accept only if the lines really agree: a solid share of the weight is explained and
            # the typical residual is small (otherwise these are plants, textures, noise)
            share = wv[inl].sum() / wv.sum()
            if share >= 0.25 and _v_dev(vc) <= 1.0:
                R_v = cand_R

    if R_v is None:
        # fallback: only the horizontals (roll), if they are clear
        if use_horizontals and len(segs):
            roll_a, hsup = _roll_from_horizontals(segs, 0.015 * max(hs, ws), split_y, max_roll)
            if roll_a is not None:
                R = cv2.Rodrigues(np.array([0.0, 0.0, -np.radians(roll_a)]))[0]
                Hs = K_small @ R @ np.linalg.inv(K_small)
                q = np.c_[_apply(Hs, hsup[:, :2]), _apply(Hs, hsup[:, 2:])]
                res = Result("ok_horizontals_only", roll_deg=roll_a, n_h_lines=len(hsup),
                             dev_h_before=_h_dev(hsup), dev_h_after=_h_dev(q),
                             yaw_note="no usable vertical lines: only the roll was leveled",
                             n_lines=len(vert))
                if abs(roll_a) > max_roll:
                    res.status = "skipped_too_tilted"
                    return res
                return _finish(res, R, full_size, focal_mm, sensor_w_mm)
        return Result("skipped_no_lines", n_lines=len(vert), n_inliers=int(inl.sum()),
                      dev_before=dev_before)

    rvec = cv2.Rodrigues(R_v)[0].ravel()
    pitch, roll = float(np.degrees(rvec[0])), float(np.degrees(rvec[2]))
    R = R_v
    yaw, n_h, note = 0.0, 0, ""
    cand_h = sup_h = None
    if use_horizontals and len(segs):
        a, cand_h, sup_h, note = estimate_yaw(segs, K_small, R_v, min_len, max_yaw, split_y)
        if a is not None:
            yaw, n_h = a, int(sup_h.sum())
            R = cv2.Rodrigues(np.array([0.0, np.radians(a), 0.0]))[0] @ R_v
    Hs = K_small @ R @ np.linalg.inv(K_small)
    vq = np.c_[_apply(Hs, vert[:, :2]), _apply(Hs, vert[:, 2:])]
    left = region_v == 0
    res = Result("ok", pitch, roll, dev_before, _v_dev(vq), len(vert), int(inl.sum()),
                 yaw_deg=yaw, n_h_lines=n_h, yaw_note=note,
                 dev_v_left=_median_or_none(np.abs(_wrap(_vertical_dev(vq[left])))),
                 dev_v_right=_median_or_none(np.abs(_wrap(_vertical_dev(vq[~left])))))
    if n_h:
        hsup = cand_h[sup_h]
        hq = np.c_[_apply(Hs, hsup[:, :2]), _apply(Hs, hsup[:, 2:])]
        res.dev_h_before, res.dev_h_after = _h_dev(hsup), _h_dev(hq)
        top = (hsup[:, 1] + hsup[:, 3]) / 2 < split_y
        for name, m in (("dev_h_top", top), ("dev_h_bottom", ~top)):
            if m.sum() >= 2:
                setattr(res, name, _h_dev(hq[m]))
    if abs(pitch) > max_pitch or abs(roll) > max_roll:
        res.status = "skipped_too_tilted"
        return res
    return _finish(res, R, full_size, focal_mm, sensor_w_mm)


def _finish(res: Result, R: np.ndarray, full_size, focal_mm, sensor_w_mm) -> Result:
    wf, hf = full_size
    K = camera_matrix(wf, hf, focal_mm, sensor_w_mm)
    H = K @ R @ np.linalg.inv(K)
    corners = np.array([[0, 0], [wf, 0], [wf, hf], [0, hf]], float)
    s, (x0, y0, rw, rh) = inscribed_rect(_apply(H, corners), wf / hf, wf, hf)
    res.homography, res.crop_scale, res.crop = H, s, (x0, y0, rw, rh)
    return res


def warp(img: np.ndarray, res: Result) -> np.ndarray:
    """Apply the estimated correction and crop to the inscribed rectangle."""
    if res.homography is None or res.crop is None:
        return img
    x0, y0, rw, rh = res.crop
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
    out = cv2.warpPerspective(img, T @ res.homography, (rw, rh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    np.clip(out, 0.0, 1.0, out=out)  # cubic interpolation can overshoot slightly below 0 / above 1
    return out
