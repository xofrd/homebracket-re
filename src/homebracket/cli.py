"""Command line: `homebracket check|run <folder>`."""
from __future__ import annotations

import argparse
import sys
import subprocess
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from . import exif, grouping, pipeline, report


def _child_rss_mb(pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL).stdout.strip()
        return int(out) / 1024 if out else 0.0
    except Exception:
        return 0.0


def _worker(q, *a):
    try:
        q.put(pipeline.process_group(*a))
    except Exception as e:  # report instead of crashing silently
        q.put({"status": f"error: {e}"})


def run_isolated(args_tuple, max_mem_mb: float):
    """Run one group in a fresh process; kill it if it exceeds the memory limit."""
    ctx = get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, *args_tuple))
    p.start()
    peak = 0.0
    while p.is_alive():
        peak = max(peak, _child_rss_mb(p.pid))
        if max_mem_mb and peak > max_mem_mb:
            p.kill()
            p.join()
            return {"status": f"stopped: memory limit {max_mem_mb:.0f} MB exceeded (seen {peak:.0f} MB)",
                    "peak_mb": round(peak)}
        time.sleep(0.25)
        if not q.empty():
            break
    try:
        r = q.get(timeout=30)
    except Exception:
        r = {"status": "error: no result from worker"}
    p.join(timeout=10)
    return r


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
    out = Path(args.output).expanduser() if args.output else Path(args.folder).expanduser() / "Out"
    out.mkdir(parents=True, exist_ok=True)
    opts = pipeline.Options(scale=args.scale, do_align=args.align, do_straighten=not args.no_straighten,
                            focal_mm=args.focal_mm, max_pitch=args.max_pitch, max_roll=args.max_roll, max_yaw=args.max_yaw,
                            use_horizontals=not args.no_horizontals, balance=args.balance,
                            quality=args.quality, tiff=args.tiff, tolerance_deg=args.tolerance)
    todo = [g for g in groups if not g.skipped]
    if args.only:
        todo = todo[: args.only]
    rows = []
    for n, g in enumerate(todo, 1):
        ordered = g.by_brightness()
        focal = args.focal_mm or ordered[0].focal_mm
        paths = [str(s.path) for s in ordered]
        print(f"({n}/{len(todo)}) processing {g.name} ...", flush=True)
        # one fresh process per group: memory is returned to the OS after each one
        r = run_isolated((paths, len(paths) // 2, focal, str(out), asdict(opts)), args.max_mem_mb)
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
            s.add_argument("-o", "--output", default=None, help="output folder (default: an \"Out\" subfolder inside the input folder)")
            s.add_argument("--scale", type=float, default=1.0, help="0.5 = half size (fast tests)")
            s.add_argument("--align", action="store_true", help="align frames (hand-held brackets)")
            s.add_argument("--no-straighten", action="store_true")
            s.add_argument("--focal-mm", type=float, help="override focal length from EXIF")
            s.add_argument("--max-pitch", type=float, default=12.0)
            s.add_argument("--max-roll", type=float, default=6.0)
            s.add_argument("--max-yaw", type=float, default=6.0, help="largest lateral turn to correct, degrees")
            s.add_argument("--no-horizontals", action="store_true", help="correct verticals only")
            s.add_argument("--balance", type=float, default=0.5, help="light balance strength 0..1 (0 = off)")
            s.add_argument("--max-mem-mb", type=float, default=5500, help="stop a group above this memory (0 = no limit)")
            s.add_argument("--tolerance", type=float, default=0.3, help="vertical tolerance in degrees")
            s.add_argument("--quality", type=int, default=95)
            s.add_argument("--tiff", action="store_true", help="also write a 16-bit TIFF (~200 MB each)")
            s.add_argument("--only", type=int, help="process only the first N groups")
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
