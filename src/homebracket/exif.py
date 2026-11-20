"""Read capture metadata with exiftool (one call for the whole folder)."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAW_EXTS = {".arw", ".dng", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"}

_TAGS = ["-ExposureTime", "-FNumber", "-ISO", "-ExposureCompensation", "-ShutterCount",
         "-SubSecDateTimeOriginal", "-DateTimeOriginal", "-LensID", "-LensModel", "-FocalLength"]


@dataclass(frozen=True)
class Shot:
    path: Path
    exposure_time: float | None = None
    f_number: float | None = None
    iso: float | None = None
    ev_bias: float | None = None
    shutter_count: int | None = None
    timestamp: float | None = None  # seconds since epoch
    lens: str | None = None
    focal_mm: float | None = None

    @property
    def stops(self) -> float | None:
        """Exposure time in stops (log2 seconds); higher = brighter at fixed aperture/ISO."""
        if not self.exposure_time or self.exposure_time <= 0:
            return None
        return math.log2(self.exposure_time)


def _parse_ts(value) -> float | None:
    if not value or not isinstance(value, str):
        return None
    m = re.match(r"(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)(\.\d+)?([+-]\d\d:\d\d)?", value)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    frac = float(m.group(7)) if m.group(7) else 0.0
    tz = timezone.utc
    if m.group(8):
        sign = 1 if m.group(8)[0] == "+" else -1
        hh, mm = int(m.group(8)[1:3]), int(m.group(8)[4:6])
        tz = timezone(sign * timedelta(hours=hh, minutes=mm))
    return datetime(y, mo, d, h, mi, s, tzinfo=tz).timestamp() + frac


def find_raws(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in RAW_EXTS and not p.name.startswith("."))


def read_shots(paths: list[Path]) -> list[Shot]:
    if shutil.which("exiftool") is None:
        raise RuntimeError("exiftool not found. Install it (macOS: `brew install exiftool`).")
    if not paths:
        return []
    out = subprocess.run(["exiftool", "-j", "-n", *_TAGS, *map(str, paths)],
                         capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout
    by_name = {Path(r["SourceFile"]).name: r for r in json.loads(out)}
    shots = []
    for p in paths:
        r = by_name.get(p.name, {})
        shots.append(Shot(
            path=p,
            exposure_time=r.get("ExposureTime"),
            f_number=r.get("FNumber"),
            iso=r.get("ISO"),
            ev_bias=r.get("ExposureCompensation"),
            shutter_count=r.get("ShutterCount"),
            timestamp=_parse_ts(r.get("SubSecDateTimeOriginal") or r.get("DateTimeOriginal")),
            lens=r.get("LensID") or r.get("LensModel"),
            focal_mm=r.get("FocalLength"),
        ))
    return shots
