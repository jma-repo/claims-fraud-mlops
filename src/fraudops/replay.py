"""Replay the newest month's claims into the API in report order, like a live FNOL stream.

In production this is a Kafka or Event Hubs consumer; for a demo, an HTTP loop is enough to
show online features, scoring latency and the dashboard's live feed.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

import httpx
import pandas as pd

from fraudops.config import Settings

DERIVED = {"month", "loss_ts", "total_loss", "total_claimed"}


def to_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in record.items():
        if key in DERIVED:
            continue
        if isinstance(value, float) and math.isnan(value):
            value = None
        elif isinstance(value, pd.Timestamp):
            value = value.isoformat()
        payload[key] = value
    return payload


def replay(settings: Settings, url: str, rate: float, limit: int = 0) -> int:
    claims = pd.read_parquet(settings.paths.replay).sort_values("report_ts")
    if limit:
        claims = claims.head(limit)
    bands: Counter[str] = Counter()
    latencies = []
    with httpx.Client(base_url=url.rstrip("/"), timeout=30.0) as client:
        for record in claims.to_dict(orient="records"):
            response = client.post("/api/score", json=to_payload(record))
            response.raise_for_status()
            result = response.json()
            bands[result["band"]] += 1
            latencies.append(result["latency_ms"])
            print(f"{result['claim_id']}  {record['loss_type']:<14} ${result['total_claimed']:>10,.0f}  "
                  f"p={result['probability']:.3f}  {result['band']}", flush=True)
            if rate > 0:
                time.sleep(1.0 / rate)
    latencies.sort()
    print(f"\n{len(latencies)} claims replayed: " + ", ".join(f"{band} {count}" for band, count in bands.most_common())
          + f". Median scoring latency {latencies[len(latencies) // 2]:.0f} ms.")
    return 0
