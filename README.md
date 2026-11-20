# HomeBracket (homebracket-re)

**Open-source automation for bracketed real estate and architectural photography.**

HomeBracket aims to turn a folder of bracketed RAW files (for example 3 shots at -2 / 0 / +2 EV) into finished, straight, color-balanced interior and exterior photos, in batch, with no paid software.

> **Status: early stage / pre-alpha.** Nothing here is production-ready yet. This repository was just born; the pipeline below is the plan, not a finished product. Expect rough edges and breaking changes.

## Why this exists

Real estate photographers often shoot 50 to 200+ photos per property, each one a bracket. Merging, balancing white, straightening verticals and correcting the lens by hand takes hours per listing. Existing automated services are paid and closed. HomeBracket is a personal project that tries to automate the repetitive part with free, open tools (OpenCV, rawpy, exiftool, darktable) so photographers can spend their time shooting.

## What it is designed for

HomeBracket is built around **architectural-style photography**:

- Camera on a tripod (or very still), **looking straight at the subject, about 90 degrees to the wall**, horizontal (landscape) frames.
- **Bracketed exposures**, typically 3 frames per scene (normal, dark, bright), shot in sequence with a timer or remote so the frames are already aligned.
- Wide and ultra-wide lenses used in tight rooms (bathrooms, laundry rooms, offices).

## Planned pipeline

Each step is a swappable module so different approaches can be compared on the same photos.

1. **Group** frames into brackets (fixed groups of 3 by default; optionally by capture time or EXIF shot counter).
2. **Align** frames (optional; off by default when shooting from a tripod with a timer).
3. **Merge** exposures (Mertens exposure fusion, Enfuse, linear RAW merge, or darktable).
4. **White balance** (camera value, automatic from walls, or a reference).
5. **Straighten** (vertical lines and horizon).
6. **Lens correction** (vignetting and chromatic aberration from embedded EXIF data when available).
7. **Export** (sRGB JPEG, 16-bit TIFF).

A test bench that renders the same scenes through different variants side by side is part of the plan, so contributors can compare results objectively.

## Prototype v0.1 (what exists today)

Implemented: bracket grouping and sanity checks (by file order or capture time), RAW development with `rawpy`, Mertens exposure fusion (OpenCV), optional alignment, automatic vertical correction (pitch and roll from vertical-line vanishing direction, with a tilt limit so angled shots are flagged instead of forced), JPEG export with copied EXIF, and an HTML before/after report. Each bracket runs in its own process to keep memory use low. Unit tests use synthetic images only; **it has not been validated on real photos yet**.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"        # needs exiftool installed (macOS: brew install exiftool)
homebracket check /path/to/raws                 # show how files are grouped, with warnings
homebracket run /path/to/raws --scale 0.5 --only 1   # quick half-size test of one bracket
homebracket run /path/to/raws                        # full resolution
```

Results are written to an `Out` subfolder inside the folder with the original RAW files (override with `-o`); the originals are never modified. Output is JPEG by default (about 10 to 20 MB at 33 MP). A 16-bit TIFF (`--tiff`) is about 200 MB per image at 33 MP, so it is opt-in.

## Working together with darktable (goal)

HomeBracket should work **alongside [darktable](https://www.darktable.org/)**, not instead of it. The idea is to automate the repetitive steps and hand the photographer a result that is already open and adjustable in darktable. Possible integration paths, to be researched and compared:

1. **darktable as a backend:** HomeBracket calls `darktable-cli` to apply a saved style / XMP sidecar and export (darktable's CLI applies stored parameters; it does not appear to run the automatic perspective fit by itself).
2. **XMP sidecars as the bridge:** HomeBracket computes values (for example the vertical correction) and writes them into an XMP history stack so the image opens in darktable already corrected. This needs reverse-engineering of darktable's module parameter format.
3. **A Lua script / plugin inside darktable:** darktable has a [Lua API](https://docs.darktable.org/usermanual/development/en/lua/api/) and a community [lua-scripts](https://github.com/darktable-org/lua-scripts) collection (for example `enfuseAdvanced` and bracket-grouping scripts), which could call HomeBracket from the darktable interface.
4. **Plain files:** HomeBracket outputs 16-bit TIFF or DNG that darktable imports as usual.

We do not know yet whether a ready-made connection exists. If you work on darktable (Lua, CLI, or modules such as perspective correction) and have ideas or prior art, **please open an issue**: help from the darktable community is explicitly wanted.

## Optional later: color grading in DaVinci Resolve (idea)

As a possible final, optional step, finished photos could be passed through **DaVinci Resolve** for color grading. Newer Resolve versions are reported to include a Photo page for still images; which edition includes it, and whether it can be scripted for batch work, has **not been verified** yet. This is an idea for later, not a commitment, and it would stay optional so the core pipeline remains free and open source.

## Known limitations (please read)

- **Angled or corner shots do not work well.** If the camera is not square to the subject (for example a corner, diagonal, or strongly tilted perspective), vertical correction and alignment can stretch and distort the image heavily, and the result is likely to look bad. HomeBracket assumes verticals should end up vertical and the view is roughly perpendicular to the main wall.
- **Hand-held brackets** may need the optional alignment step and can still produce ghosting or soft results.
- **Vertical (portrait) frames** are not a target use case.
- Lens data is read from EXIF where the lens reports it. Lenses that report nothing may need a manual profile.
- Everything here is **untested at scale**. The first reference camera is a Sony A7 IV with an ultra-wide full-frame lens.

## Contributing

This is a personal project and **contributions are very welcome**: code, test photos, bug reports, ideas, documentation, translations, comparisons with other tools. If you shoot real estate or architecture, your sample bracket sets (that you have the right to share) are especially valuable.

- Open an issue to discuss an idea or report a problem.
- Pull requests are welcome; small focused ones are easiest to review.
- Please only share photos you own or have permission to share, and avoid images with people or identifiable private details.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE).
