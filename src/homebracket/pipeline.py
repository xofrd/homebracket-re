"""One bracket group in, one finished image out. Runs in its own process (memory is freed)."""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import balance, export, fuse, straighten


@dataclass
class Options:
    scale: float = 1.0
    do_align: bool = False
    do_straighten: bool = True
    focal_mm: float | None = None
    max_pitch: float = 12.0
    max_roll: float = 6.0
    max_yaw: float = 6.0
    use_horizontals: bool = True
    balance: float = 0.5
    roi: tuple | None = None
    quality: int = 95
    tiff: bool = False
    tolerance_deg: float = 0.3


def _thumb(img: np.ndarray, path: Path, width: int = 1000) -> None:
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    export.write_jpeg(small, path, 85)


def _peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def process_group(shots_paths: list[str], ref_index: int, focal_mm: float | None,
                  out_dir: str, opts_dict: dict) -> dict:
    """shots_paths: raw files ordered dark -> bright; ref_index: the middle exposure."""
    opts = Options(**opts_dict)
    t0 = time.time()
    out = Path(out_dir)
    thumbs = out / "report" / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    ref_path = Path(shots_paths[ref_index])
    stem = ref_path.stem

    images = [fuse.develop(Path(p), opts.scale) for p in shots_paths]
    if opts.do_align:
        images = fuse.align(images)
    _thumb(images[ref_index], thumbs / f"{stem}_before.jpg")
    fused = fuse.fuse(images)
    del images

    info = {"status": "not_straightened"}
    if opts.do_straighten:
        h, w = fused.shape[:2]
        sw = 1600
        gray = cv2.cvtColor(cv2.resize(fused, (sw, int(h * sw / w)), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY)
        gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
        res = straighten.estimate(gray, (w, h), focal_mm or 10.0,
                                  opts.max_pitch, opts.max_roll, opts.max_yaw, opts.use_horizontals,
                                  roi=opts.roi)
        fused = straighten.warp(fused, res)
        info = {k: v for k, v in asdict(res).items() if k not in ("homography",)}
        worst = res.worst_deg if res.worst_deg is not None else res.dev_after
        info["worst_deg"] = None if worst is None else round(worst, 3)
        if res.status == "ok" and worst is not None and worst > opts.tolerance_deg:
            info["status"] = "review_tolerance"

    if opts.balance > 0:
        fused, bsum = balance.apply(fused, opts.balance)
        info["balance_min_stops"] = round(bsum.get("min_stops", 0.0), 2)
        info["balance_max_stops"] = round(bsum.get("max_stops", 0.0), 2)

    jpg = out / f"{stem}_hb.jpg"
    export.write_jpeg(fused, jpg, opts.quality)
    export.copy_exif(ref_path, jpg)
    if opts.tiff:
        export.write_tiff16(fused, out / f"{stem}_hb.tif")
    _thumb(fused, thumbs / f"{stem}_after.jpg")
    return {"stem": stem, "output": str(jpg), "seconds": round(time.time() - t0, 1),
            "peak_mb": round(_peak_mb()), "size": list(fused.shape[1::-1]), **info}
