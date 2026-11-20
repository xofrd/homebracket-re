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
