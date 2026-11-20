"""Write the final image (JPEG by default; 16-bit TIFF optional) and copy basic EXIF."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

_COPY = ["Make", "Model", "LensModel", "LensID", "FocalLength", "FNumber", "ISO", "ExposureTime",
         "DateTimeOriginal", "OffsetTimeOriginal", "SubSecTimeOriginal"]


def write_jpeg(img: np.ndarray, path: Path, quality: int = 95) -> None:
    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(path, "JPEG", quality=quality, subsampling=0, optimize=True)


def write_tiff16(img: np.ndarray, path: Path) -> None:
    u16 = np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    tifffile.imwrite(path, u16, photometric="rgb", compression="deflate")


def copy_exif(src_raw: Path, dst: Path) -> None:
    if shutil.which("exiftool") is None:
        return
    args = ["exiftool", "-q", "-overwrite_original", "-TagsFromFile", str(src_raw),
            *[f"-{t}" for t in _COPY], "-Software=HomeBracket", "-Orientation#=1", str(dst)]
    subprocess.run(args, check=False, stdin=subprocess.DEVNULL, capture_output=True)
