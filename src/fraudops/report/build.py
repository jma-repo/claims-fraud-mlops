"""Assemble every report into one dashboard page.

The page embeds its data, so it opens from disk or GitHub Pages with no server. When it is
served by the API, it detects that and switches on the live parts (copilot chat, scoring,
the FNOL feed).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fraudops.config import Settings

STATIC = Path(__file__).parent / "static"
QUEUE_SIZE = 30


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _memberships(rings: dict[str, Any], segments: dict[str, Any]) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    for ring in rings.get("rings", [])[:5]:
        for claim_id in ring["claim_ids"]:
            tags.setdefault(claim_id, []).append(ring["id"])
    for pattern in segments.get("emerging", []):
        for claim_id in pattern["claim_ids"]:
            tags.setdefault(claim_id, []).append(pattern["id"])
    return tags


def _queue(scored: pd.DataFrame, tags: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = []
    for row in scored.sort_values("queue_rank").head(QUEUE_SIZE).itertuples(index=False):
        rows.append({
            "claim_id": row.claim_id, "rank": int(row.queue_rank), "band": row.band,
            "probability": float(row.probability), "expected_value": float(row.expected_value),
            "total_claimed": float(row.total_claimed), "loss_type": row.loss_type, "city": row.city,
            "province": row.province, "reported": f"{pd.Timestamp(row.report_ts):%Y-%m-%d}",
            "reasons_for": json.loads(row.reasons_for), "reasons_against": json.loads(row.reasons_against),
            "narrative": row.narrative, "tags": tags.get(row.claim_id, []),
            "facts": {
                "loss_hour": int(row.loss_hour), "report_delay_days": int(row.report_delay_days),
                "policy_tenure_days": int(row.policy_tenure_days), "n_occupants": int(row.n_occupants),
                "n_injured": int(row.n_injured), "soft_tissue_only": bool(row.soft_tissue_only),
                "police_report": bool(row.police_report), "witnesses": int(row.witnesses),
                "prior_claims_3y": int(row.prior_claims_3y), "vehicle_value": float(row.vehicle_value),
                "repair_estimate": round(math.expm1(row.log_repair_estimate), 2),
                "injury_reserve": round(math.expm1(row.log_injury_reserve), 2), "tow_fee": float(row.tow_fee),
                "storage_days": int(row.storage_days), "tow_distance_km": float(row.tow_distance_km),
                "body_shop_id": row.body_shop_id, "clinic_id": row.clinic_id, "rep_id": row.rep_id,
                "tow_company_id": row.tow_company_id,
            },
        })
    for row in rows:  # NaN provider ids become null in JSON
        row["facts"] = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row["facts"].items()}
    return rows


def _columnar(points: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """The pattern map has thousands of points; columns keep the embedded JSON small."""
    keys = ["x", "y", "claim_id", "segment", "current", "loss_type", "total_claimed", "probability",
            "confirmed_fraud", "emerging", "in_queue"]
    return {key: [point.get(key) for point in points] for key in keys}


def collect(settings: Settings) -> dict[str, Any]:
    paths = settings.paths
    reports = paths.reports
    scored = pd.read_parquet(paths.scored)
    segments = _read(reports / "segments.json", {})
    rings = _read(reports / "rings.json", {"rings": [], "networks": []})
    tags = _memberships(rings, segments)
    rag_eval = _read(reports / "rag_eval.json", {})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "month": settings.pipeline["data"]["new_month"],
        "business": settings.business,
        "metrics": _read(reports / "metrics.json"),
        "latest": _read(reports / "latest_month.json"),
        "queue": _queue(scored, tags),
        "segments": {k: v for k, v in segments.items() if k != "map"},
        "map": _columnar(segments.get("map", [])),
        "rings": {"rings": rings["rings"][:5], "networks": rings["networks"]},
        "drift": _read(reports / "drift.json"),
        "titles": _read(reports / "pattern_titles.json", {}),
        "rag_eval": {"mode": rag_eval.get("mode"), "metrics": rag_eval.get("metrics", {}),
                     "rows": [{k: r[k] for k in ("id", "question", "rank", "declined", "mode", "elapsed_ms", "out_of_scope")}
                              for r in rag_eval.get("rows", [])]},
        "copilot_examples": _read(reports / "copilot_examples.json", []),
        "kb": _read(paths.kb_index.with_suffix(".json"), {}),
    }


def run(settings: Settings) -> Path:
    from plotly.offline import get_plotlyjs

    paths = settings.paths
    site = paths.site
    assets = site / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("app.js", "styles.css", "favicon.svg"):
        shutil.copyfile(STATIC / name, assets / name)
    plotly_js = assets / "plotly.min.js"
    bundled = get_plotlyjs().encode("utf-8")
    if not plotly_js.exists() or plotly_js.read_bytes() != bundled:
        plotly_js.write_bytes(bundled)  # bytes: text mode on Windows would add CRLF to a vendored file

    data = collect(settings)
    environment = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"]))
    page = environment.get_template("template.html").render(
        # "</" cannot appear inside the inline JSON block, so a string in the data cannot close it.
        data_json=json.dumps(data, separators=(",", ":"), default=str).replace("</", "<\\/"),
        month=data["month"],
        generated_at=data["generated_at"],
        # Content hashes in asset URLs: a redeploy reaches visitors at once instead of after the CDN cache expires.
        v={name: hashlib.sha256((assets / name).read_bytes()).hexdigest()[:10]
           for name in ("app.js", "styles.css", "plotly.min.js")},
    )
    (site / "index.html").write_text(page, encoding="utf-8", newline="\n")
    (site / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: serve assets as-is
    return site / "index.html"
