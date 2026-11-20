"""Group shots into brackets and sanity-check them (never guess silently)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .exif import Shot


def _natural(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


@dataclass
class Group:
    shots: list[Shot]
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def name(self) -> str:
        return f"{self.shots[0].path.stem}-{self.shots[-1].path.stem}"

    def by_brightness(self) -> list[Shot]:
        """Shots ordered dark -> bright by exposure time (not by file order)."""
        return sorted(self.shots, key=lambda s: (s.exposure_time or 0))

    @property
    def reference(self) -> Shot:
        """The middle exposure (the 'normal' frame)."""
        return self.by_brightness()[len(self.shots) // 2]


def validate(group: Group, expected_span=(2.5, 5.5)) -> Group:
    s = group.shots
    if any(x.exposure_time is None for x in s):
        group.warnings.append("missing exposure time in EXIF")
        return group
    if len({(x.f_number, x.iso) for x in s}) > 1:
        group.warnings.append("aperture/ISO differ inside the group (is this really one bracket?)")
    if len(s) == 3:
        dark, mid, bright = (x.stops for x in group.by_brightness())
        span = bright - dark
        if not (expected_span[0] <= span <= expected_span[1]):
            group.warnings.append(f"bracket spread {span:.1f} stops (expected about 4 for -2/0/+2)")
        if abs((mid - dark) - (bright - mid)) > 1.0:
            group.warnings.append("middle exposure is not centred between dark and bright")
    counts = [x.shutter_count for x in s]
    if None not in counts and sorted(counts) != list(range(min(counts), min(counts) + len(counts))):
        group.warnings.append("shutter counts are not consecutive (frames may belong to different scenes)")
    ts = [x.timestamp for x in s if x.timestamp is not None]
    if len(ts) == len(s) and max(ts) - min(ts) > 3.0:
        group.warnings.append(f"frames span {max(ts) - min(ts):.1f} s (slow bracket or different scenes)")
    return group


def group_by_name(shots: list[Shot], size: int = 3) -> list[Group]:
    shots = sorted(shots, key=lambda s: _natural(s.path))
    groups = []
    for i in range(0, len(shots), size):
        chunk = shots[i:i + size]
        g = Group(chunk)
        if len(chunk) < size:
            g.warnings.append(f"incomplete group ({len(chunk)} of {size} frames): skipped")
            g.skipped = True
        else:
            validate(g)
        groups.append(g)
    return groups


def group_by_time(shots: list[Shot], gap: float = 4.0) -> list[Group]:
    if any(s.timestamp is None for s in shots):
        raise ValueError("--group-by time needs a capture timestamp on every file")
    shots = sorted(shots, key=lambda s: (s.timestamp, _natural(s.path)))
    groups, cur = [], [shots[0]] if shots else []
    for prev, s in zip(shots, shots[1:]):
        if s.timestamp - prev.timestamp > gap:
            groups.append(validate(Group(cur)))
            cur = []
        cur.append(s)
    if cur:
        groups.append(validate(Group(cur)))
    return groups
