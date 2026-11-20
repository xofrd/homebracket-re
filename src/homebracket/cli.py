"""Command line: `homebracket check|run <folder>`."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from . import exif, grouping, pipeline, report


def _groups(args):
    folder = Path(args.folder).expanduser()
    paths = exif.find_raws(folder)
    if not paths:
        sys.exit(f"No RAW files found in {folder}")
    shots = exif.read_shots(paths)
    if args.group_by == "time":
        return grouping.group_by_time(shots, args.time_gap)
    return grouping.group_by_name(shots, args.group_size)


def _show(groups):
    for i, g in enumerate(groups, 1):
        order = " ".join(f"{s.path.name}({s.ev_bias:+g}EV,1/{round(1/s.exposure_time)})"
                         if s.exposure_time and s.ev_bias is not None else s.path.name for s in g.shots)
        print(f"[{i}] {g.name}: {order}")
        for w in g.warnings:
            print(f"    ! {w}")


def cmd_check(args):
    _show(_groups(args))


def cmd_run(args):
    groups = _groups(args)
    _show(groups)
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    opts = pipeline.Options(scale=args.scale, do_align=args.align, do_straighten=not args.no_straighten,
                            focal_mm=args.focal_mm, max_pitch=args.max_pitch, max_roll=args.max_roll,
                            quality=args.quality, tiff=args.tiff, tolerance_deg=args.tolerance)
    todo = [g for g in groups if not g.skipped]
    if args.only:
        todo = todo[: args.only]
    rows = []
    ctx = get_context("spawn")
    for n, g in enumerate(todo, 1):
        ordered = g.by_brightness()
        focal = args.focal_mm or ordered[0].focal_mm
        paths = [str(s.path) for s in ordered]
        print(f"({n}/{len(todo)}) processing {g.name} ...", flush=True)
        # one fresh process per group: memory is returned to the OS after each one
        with ProcessPoolExecutor(1, mp_context=ctx) as ex:
            try:
                r = ex.submit(pipeline.process_group, paths, len(paths) // 2, focal,
                              str(out), asdict(opts)).result()
            except Exception as e:  # keep going with the other groups
                r = {"status": f"error: {e}"}
        r.update(name=g.name, warnings=g.warnings)
        rows.append(r)
        print(f"    -> {r.get('status')}  {r.get('seconds', '-')} s  peak {r.get('peak_mb', '-')} MB", flush=True)
    path = report.write_report(out, rows)
    print(f"Report: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="homebracket", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("folder")
        s.add_argument("--group-by", choices=["name", "time"], default="name")
        s.add_argument("--group-size", type=int, default=3)
        s.add_argument("--time-gap", type=float, default=4.0, help="seconds between brackets (group-by time)")
        s.set_defaults(fn=fn)
        if name == "run":
            s.add_argument("-o", "--output", default="out")
            s.add_argument("--scale", type=float, default=1.0, help="0.5 = half size (fast tests)")
            s.add_argument("--align", action="store_true", help="align frames (hand-held brackets)")
            s.add_argument("--no-straighten", action="store_true")
            s.add_argument("--focal-mm", type=float, help="override focal length from EXIF")
            s.add_argument("--max-pitch", type=float, default=12.0)
            s.add_argument("--max-roll", type=float, default=6.0)
            s.add_argument("--tolerance", type=float, default=0.3, help="vertical tolerance in degrees")
            s.add_argument("--quality", type=int, default=95)
            s.add_argument("--tiff", action="store_true", help="also write a 16-bit TIFF (~200 MB each)")
            s.add_argument("--only", type=int, help="process only the first N groups")
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
