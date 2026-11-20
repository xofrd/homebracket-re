"""Side-by-side HTML report: before/after, metrics, warnings."""
from __future__ import annotations

import html
from pathlib import Path

_CSS = """body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
.g{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 20px}
.p{display:flex;gap:12px;flex-wrap:wrap}.p figure{margin:0;flex:1;min-width:300px}
img{width:100%;height:auto;border-radius:4px}table{border-collapse:collapse;font-size:13px}
td,th{padding:3px 10px;border-bottom:1px solid #eee;text-align:left}.warn{color:#b45309}
.ok{color:#15803d}.bad{color:#b91c1c}"""


def _fmt(v, digits=2):
    return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def write_report(out_dir: Path, rows: list[dict]) -> Path:
    parts = [f"<!doctype html><meta charset=utf-8><title>HomeBracket report</title><style>{_CSS}</style>",
             "<h1>HomeBracket report</h1>"
             "<p>Vertical deviation = median angle of near-vertical lines from true vertical. "
             "It is measured on the same lines used to correct, so treat it as a sanity check; "
             "judge the result with your eyes.</p>"]
    for r in rows:
        cls = "ok" if r.get("status") == "ok" else "bad"
        parts.append(f"<div class=g><h3>{html.escape(r['name'])} <small class={cls}>{html.escape(str(r.get('status')))}</small></h3>")
        if "stem" in r:
            parts.append(f"<div class=p><figure><figcaption>before (middle exposure)</figcaption>"
                         f"<img src='report/thumbs/{r['stem']}_before.jpg'></figure>"
                         f"<figure><figcaption>after (fused + straightened)</figcaption>"
                         f"<img src='report/thumbs/{r['stem']}_after.jpg'></figure></div>")
            parts.append("<table>"
                         f"<tr><th>time</th><td>{_fmt(r.get('seconds'),1)} s</td><th>peak RAM</th><td>{_fmt(r.get('peak_mb'))} MB</td></tr>"
                         f"<tr><th>vertical dev. before</th><td>{_fmt(r.get('dev_before'))}°</td><th>after</th><td>{_fmt(r.get('dev_after'))}°</td></tr>"
                         f"<tr><th>pitch / roll</th><td>{_fmt(r.get('pitch_deg'),1)}° / {_fmt(r.get('roll_deg'),1)}°</td><th>crop kept</th><td>{_fmt(r.get('crop_scale'))}</td></tr>"
                         f"<tr><th>yaw corrected</th><td>{_fmt(r.get('yaw_deg'),1)}° ({_fmt(r.get('n_h_lines'))} lines)</td><th>horizontal dev. before / after</th><td>{_fmt(r.get('dev_h_before'))}° / {_fmt(r.get('dev_h_after'))}°</td></tr>"
                         f"<tr><th>light balance (stops)</th><td>{_fmt(r.get('balance_min_stops'))} to +{_fmt(r.get('balance_max_stops'))}</td><th>note</th><td>{html.escape(str(r.get('yaw_note') or ''))}</td></tr>"
                         f"<tr><th>residual: left / right (verticals)</th><td>{_fmt(r.get('dev_v_left'))}° / {_fmt(r.get('dev_v_right'))}°</td><th>top / bottom (horizontals)</th><td>{_fmt(r.get('dev_h_top'))}° / {_fmt(r.get('dev_h_bottom'))}°</td></tr>"
                         f"<tr><th>lines / inliers</th><td>{_fmt(r.get('n_lines'))} / {_fmt(r.get('n_inliers'))}</td><th>output</th><td>{html.escape(Path(r['output']).name)}</td></tr></table>")
        for w in r.get("warnings", []):
            parts.append(f"<p class=warn>⚠ {html.escape(w)}</p>")
        parts.append("</div>")
    path = out_dir / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
