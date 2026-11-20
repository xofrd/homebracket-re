from pathlib import Path

import cv2
import numpy as np
import pytest

from homebracket import fuse, grouping, straighten
from homebracket.exif import Shot


def synthetic_wall(w=1600, h=1066):
    """Fronto-parallel 'room': vertical door/window frames and horizontal edges."""
    img = np.full((h, w, 3), 200, np.uint8)
    rng = np.random.default_rng(1)
    for x in range(100, w, 130):                       # vertical frames of different heights
        y0, y1 = int(rng.integers(0, 200)), int(rng.integers(h - 250, h))
        cv2.line(img, (x, y0), (x, y1), (20, 20, 20), 4)
        cv2.rectangle(img, (x + 20, y0 + 60), (x + 100, y0 + 160), (90, 90, 90), 3)
    for y in (180, 520, 900):
        cv2.line(img, (0, y), (w, y), (40, 40, 40), 4)
    return img


def tilted(img, pitch_deg, roll_deg, focal_mm=10.0):
    h, w = img.shape[:2]
    K = straighten.camera_matrix(w, h, focal_mm)
    rvec = np.radians([pitch_deg, 0.0, roll_deg])
    R = cv2.Rodrigues(rvec)[0]
    return cv2.warpPerspective(img, K @ R @ np.linalg.inv(K), (w, h), borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("pitch,roll", [(5.0, 1.5), (-4.0, -2.0), (0.0, 3.0), (8.0, 0.0)])
def test_straighten_recovers_synthetic_tilt(pitch, roll):
    img = tilted(synthetic_wall(), pitch, roll)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = straighten.estimate(gray, (gray.shape[1], gray.shape[0]), 10.0)
    assert res.status == "ok", res
    assert res.dev_before > 0.5
    assert res.dev_after < 0.3
    assert abs(abs(res.pitch_deg) - abs(pitch)) < 0.6
    assert abs(abs(res.roll_deg) - abs(roll)) < 0.6
    out = straighten.warp(img.astype(np.float32) / 255, res)
    assert out.shape[1] / out.shape[0] == pytest.approx(1600 / 1066, rel=0.01)
    assert 0.5 < res.crop_scale <= 1.0


def test_level_image_is_left_alone():
    img = synthetic_wall()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = straighten.estimate(gray, (1600, 1066), 10.0)
    assert res.status == "ok"
    assert abs(res.pitch_deg) < 0.3 and abs(res.roll_deg) < 0.3
    assert res.crop_scale > 0.97


def test_too_tilted_is_skipped_not_forced():
    img = tilted(synthetic_wall(), 15.0, 0.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = straighten.estimate(gray, (1600, 1066), 10.0, max_pitch=12.0)
    assert res.status == "skipped_too_tilted"
    assert res.homography is None


def test_blank_image_has_no_lines():
    gray = np.full((800, 1200), 128, np.uint8)
    assert straighten.estimate(gray, (1200, 800), 10.0).status == "skipped_no_lines"


def test_mertens_fuses_three_exposures():
    base = np.random.default_rng(0).random((200, 300, 3)).astype(np.float32)
    ims = [np.clip(base * 0.4, 0, 1), base, np.clip(base * 1.8, 0, 1)]
    out = fuse.fuse(ims)
    assert out.shape == base.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_align_undoes_a_shift():
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((530, 800, 3)).astype(np.float32), (0, 0), 6)
    base = (base - base.min()) / (base.max() - base.min())
    shifted = np.roll(base, (6, -9), axis=(0, 1))
    out = fuse.align([base, base, shifted])  # reference = middle (index 1)
    assert np.abs(out[2][20:-20, 20:-20] - base[20:-20, 20:-20]).mean() < 0.02


def _shot(name, t, ev=0.0, count=None, ts=0.0, iso=400):
    return Shot(Path(name), exposure_time=t, f_number=8.0, iso=iso, ev_bias=ev, shutter_count=count, timestamp=ts)


def test_group_by_name_orders_by_exposure_not_filename():
    shots = [_shot("DSC1.ARW", 1 / 200, 0, 1), _shot("DSC2.ARW", 1 / 750, -2, 2), _shot("DSC3.ARW", 1 / 45, 2, 3)]
    g = grouping.group_by_name(shots, 3)
    assert len(g) == 1 and g[0].warnings == []
    assert [s.path.name for s in g[0].by_brightness()] == ["DSC2.ARW", "DSC1.ARW", "DSC3.ARW"]
    assert g[0].reference.path.name == "DSC1.ARW"


def test_incomplete_group_is_skipped_with_warning():
    shots = [_shot(f"DSC{i}.ARW", 1 / 100, 0, i) for i in range(1, 5)]
    g = grouping.group_by_name(shots, 3)
    assert not g[0].skipped and g[1].skipped and g[1].warnings


def test_natural_sort_across_digit_boundaries():
    shots = [_shot("DSC10.ARW", 1 / 100, count=10), _shot("DSC9.ARW", 1 / 100, count=9)]
    g = grouping.group_by_name(shots, 2)
    assert g[0].shots[0].path.name == "DSC9.ARW"


def test_warns_when_frames_look_unrelated():
    shots = [_shot("a.ARW", 1 / 100, count=1), _shot("b.ARW", 1 / 100, count=5, ts=60), _shot("c.ARW", 1 / 100, count=9)]
    w = grouping.group_by_name(shots, 3)[0].warnings
    assert any("spread" in x for x in w) and any("consecutive" in x for x in w)


def test_group_by_time_splits_on_gap():
    shots = [_shot("a", 1 / 100, ts=0), _shot("b", 1 / 100, ts=0.4), _shot("c", 1 / 100, ts=10), _shot("d", 1 / 100, ts=10.3)]
    assert [len(g.shots) for g in grouping.group_by_time(shots, 4.0)] == [2, 2]
