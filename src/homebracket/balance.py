"""Gentle light balancing: even out broad left/right/top/bottom brightness differences.

Measures the low-frequency luminance of the image, compares it with the tone of the middle of the
frame, and applies a smooth gain map (like a soft graduated filter) that brightens dark regions and
slightly lowers very bright ones. Highlights are protected when brightening. Strength 0 = off.
"""
from __future__ import annotations

import cv2
import numpy as np

_GAMMA = 2.2


def gain_map(img: np.ndarray, strength: float, max_lift: float = 0.7, max_drop: float = 0.5):
    """Return a small gain map in stops (linear light) and its summary."""
    h, w = img.shape[:2]
    sw = 320
    small = cv2.resize(img, (sw, max(1, int(h * sw / w))), interpolation=cv2.INTER_AREA)
    y = 0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]
    logy = np.log2(np.clip(y, 1e-3, 1.0) ** _GAMMA)
    sigma = 0.18 * sw
    base = cv2.GaussianBlur(logy, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    ch, cw = base.shape
    target = float(np.median(base[ch // 4: 3 * ch // 4, cw // 4: 3 * cw // 4]))
    g = np.clip(strength * (target - base), -max_drop, max_lift)
    g = cv2.GaussianBlur(g, (0, 0), sigma * 0.3, borderType=cv2.BORDER_REFLECT)
    return g.astype(np.float32), {"min_stops": float(g.min()), "max_stops": float(g.max())}


def apply(img: np.ndarray, strength: float = 0.5, band: int = 512):
    """Apply the balancing in place (row bands keep peak memory low). Returns (img, summary)."""
    if strength <= 0:
        return img, {}
    h, w = img.shape[:2]
    g_small, summary = gain_map(img, strength)
    gain = cv2.resize(g_small, (w, h), interpolation=cv2.INTER_CUBIC)
    for r0 in range(0, h, band):
        r1 = min(h, r0 + band)
        blk = np.clip(img[r0:r1], 0.0, 1.0)
        gb = gain[r0:r1]
        peak = blk.max(axis=2)
        t = np.clip((peak - 0.75) / 0.25, 0.0, 1.0)
        prot = t * t * (3 - 2 * t)                    # smoothstep: protect highlights
        gb = np.where(gb > 0, gb * (1.0 - prot), gb)
        lin = np.power(blk, _GAMMA)
        lin *= np.exp2(gb)[..., None]
        np.clip(lin, 0.0, 1.0, out=lin)
        img[r0:r1] = np.power(lin, 1.0 / _GAMMA)
    return img, summary
