"""Develop RAW frames and fuse the bracket (Mertens exposure fusion)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rawpy


def develop(path: Path, scale: float = 1.0) -> np.ndarray:
    """RAW -> float32 RGB in [0, 1], camera white balance, sRGB, no auto brightness.

    scale <= 0.5 uses the fast half-size path (no demosaic); other scales demosaic at full size
    and then resize. Orientation is taken from the file.
    """
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                              half_size=scale <= 0.5)
    img = rgb.astype(np.float32) / 65535.0
    del rgb
    done = 0.5 if scale <= 0.5 else 1.0
    if scale < done - 1e-6 or (done == 1.0 and scale < 1.0 - 1e-6):
        f = scale / done
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def fuse(images: list[np.ndarray]) -> np.ndarray:
    """Mertens exposure fusion. Input/output float32 RGB in [0, 1]."""
    merge = cv2.createMergeMertens()
    out = merge.process([np.ascontiguousarray(i, dtype=np.float32) for i in images])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def align(images: list[np.ndarray]) -> list[np.ndarray]:
    """Optional translation alignment (median threshold bitmap) for hand-held brackets.

    Images are ordered dark -> bright; the middle exposure is the reference.
    """
    mtb = cv2.createAlignMTB()
    u8 = [cv2.cvtColor(np.clip(i * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for i in images]
    ref = len(images) // 2
    h, w = images[0].shape[:2]
    out = []
    for k, img in enumerate(images):
        if k == ref:
            out.append(img)
            continue
        dx, dy = mtb.calculateShift(u8[ref], u8[k])
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out
