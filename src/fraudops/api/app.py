"""Scoring API, investigator copilot API, Prometheus metrics, and the dashboard.

Run with `fraudops serve` (or the Docker image). Everything it needs is in `artifacts/` and
`site/`, so a fresh clone serves without retraining.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraudops import business
from fraudops.config import Settings
from fraudops.config import settings as load_settings
from fraudops.features import LOSS_TYPES, OnlineFeatureStore, derive_claim_fields, feature_sql
from fraudops.model import ModelBundle
from fraudops.rag.copilot import ClaimDirectory, Copilot
from fraudops.rag.llm import LocalLLM, llm_disabled
from fraudops.rag.retrieval import KnowledgeBase

ID = r"^[A-Za-z0-9-]{1,32}$"


class ClaimIn(BaseModel):
    """A first-notice-of-loss record: the same contract the training data follows."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str | None = Field(default=None, pattern=ID)
    report_ts: datetime
    province: Literal["ON", "AB", "NS", "NB"]
    city: str = Field(min_length=1, max_length=40)
    loss_type: Literal[tuple(LOSS_TYPES)]  # type: ignore[valid-type]
    loss_hour: int = Field(ge=0, le=23)
    report_delay_days: int = Field(ge=0, le=3650)
    policy_tenure_days: int = Field(ge=0, le=36500)
    policy_id: str | None = Field(default=None, pattern=ID)
    vehicle_value: float = Field(gt=0, le=1_000_000)
    vehicle_age: int = Field(ge=0, le=60)
    n_occupants: int = Field(ge=1, le=60)
    n_injured: int = Field(ge=0, le=60)
    soft_tissue_only: bool
    injury_reserve: float = Field(ge=0, le=10_000_000)
    police_report: bool
    witnesses: int = Field(ge=0, le=50)
    claimant_age: int | None = Field(default=None, ge=16, le=110)
    prior_claims_3y: int = Field(ge=0, le=50)
    body_shop_id: str | None = Field(default=None, pattern=ID)
    clinic_id: str | None = Field(default=None, pattern=ID)
    rep_id: str | None = Field(default=None, pattern=ID)
    tow_company_id: str | None = Field(default=None, pattern=ID)
    tow_distance_km: float = Field(ge=0, le=2000)
    storage_days: int = Field(ge=0, le=3650)
    tow_fee: float = Field(ge=0, le=1_000_000)
    repair_estimate: float = Field(ge=0, le=10_000_000)
    phone_id: str = Field(pattern=ID)
    address_id: str = Field(pattern=ID)
    narrative: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def injured_within_occupants(self) -> ClaimIn:
        if self.n_injured > self.n_occupants:
            raise ValueError("n_injured cannot exceed n_occupants")
        return self


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=500)
    claim_id: str | None = Field(default=None, pattern=ID)


# --- metrics ---------------------------------------------------------------------------------

REGISTRY = CollectorRegistry()
REQUESTS = Counter("fraudops_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram("fraudops_request_latency_seconds", "API latency", ["endpoint"], registry=REGISTRY,
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
SCORED = Counter("fraudops_scored_claims_total", "Claims scored online", ["band"], registry=REGISTRY)
PROBABILITY = Histogram("fraudops_fraud_probability", "Calibrated fraud probability of scored claims", registry=REGISTRY,
                        buckets=(0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0))
COPILOT = Counter("fraudops_copilot_answers_total", "Copilot answers", ["mode"], registry=REGISTRY)
FEATURE_PSI = Gauge("fraudops_feature_psi", "PSI of the newest month vs the last 12 months", ["feature"], registry=REGISTRY)
FEATURE_TAIL = Gauge("fraudops_feature_tail_ratio", "Rate above last year's 99th percentile vs usual", ["feature"],
                     registry=REGISTRY)
DRIFT_ALERTS = Gauge("fraudops_drift_alerts", "Features in alert under the pipeline's drift rule", registry=REGISTRY)
MODEL_INFO = Gauge("fraudops_model_info", "Model being served", ["version", "candidate", "run_id"], registry=REGISTRY)


# --- state -------------------------------------------------------------------------------------

@dataclass
class State:
    settings: Settings
    policy: business.SiuPolicy
    bundle: ModelBundle
    store: OnlineFeatureStore
    scored: pd.DataFrame
    queue_cutoff: float
    copilot: Copilot
    llm: LocalLLM | None
    live: deque = field(default_factory=lambda: deque(maxlen=200))
    lock: threading.Lock = field(default_factory=threading.Lock)


def build_state(s: Settings | None = None) -> State:
    s = s or load_settings()
    paths = s.paths
    policy = business.SiuPolicy.from_settings(s)
    bundle = ModelBundle.load(paths.model_dir)
    scored = pd.read_parquet(paths.scored)
    queue = scored[scored["in_queue"]]
    llm = None if llm_disabled() else LocalLLM()
    copilot = Copilot(KnowledgeBase.load(paths.kb_index), llm, ClaimDirectory.from_artifacts(paths.scored, paths.reports))
    drift_path = paths.reports / "drift.json"
    if drift_path.exists():
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        for row in drift["features"]:
            FEATURE_PSI.labels(row["feature"]).set(row["psi"])
            if row.get("tail_ratio") is not None:
                FEATURE_TAIL.labels(row["feature"]).set(row["tail_ratio"])
        DRIFT_ALERTS.set(len(drift.get("alerts", [])))
    meta = bundle.metadata
    MODEL_INFO.labels(str(meta.get("version")), str(meta.get("candidate")), str(meta.get("mlflow_run_id"))).set(1)
    return State(settings=s, policy=policy, bundle=bundle,
                 store=OnlineFeatureStore.from_parquet(paths.history, feature_sql(s)), scored=scored,
                 queue_cutoff=float(queue["expected_value"].min()) if len(queue) else 0.0, copilot=copilot, llm=llm)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = json.loads(frame.to_json(orient="records", date_format="iso"))
    for row in out:
        for key in ("reasons_for", "reasons_against"):
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
    return out


QUEUE_FIELDS = ["claim_id", "queue_rank", "band", "probability", "expected_value", "total_claimed", "loss_type", "city",
                "province", "report_ts", "reasons_for", "reasons_against", "narrative", "in_queue"]


def create_app(state_factory=build_state) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.fraudops = state_factory()
        yield

    app = FastAPI(title="Claims fraud triage API", version="0.1.0", lifespan=lifespan,
                  description="Scores first-notice-of-loss claims for SIU triage and answers investigator questions "
                              "from local documents. Synthetic data; a PoC.")

    def state() -> State:
        return app.state.fraudops

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "static") if route else "static"
        if endpoint.startswith("/api") or endpoint == "/metrics":
            REQUESTS.labels(endpoint, str(response.status_code)).inc()
            LATENCY.labels(endpoint).observe(time.perf_counter() - started)
        # The dashboard only talks to this origin; Plotly needs inline styles.
        response.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self'; style-src 'self' "
                                                       "'unsafe-inline'; img-src 'self' data:; connect-src 'self'")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        st = state()
        meta = st.bundle.metadata
        return {"status": "ok", "model": {"name": meta.get("model_name"), "version": meta.get("version"),
                                          "candidate": meta.get("candidate"), "mlflow_run_id": meta.get("mlflow_run_id"),
                                          "trained_at": meta.get("trained_at")},
                "history_claims": st.store.size(), "queue_cutoff": round(st.queue_cutoff, 2),
                "copilot": st.llm.status() if st.llm else {"ready": False, "reason": "LLM disabled"},
                "knowledge_base": st.copilot.kb.meta}

    @app.post("/api/score")
    def score(claim: ClaimIn) -> dict[str, Any]:
        st = state()
        started = time.perf_counter()
        record = claim.model_dump()
        record["claim_id"] = record["claim_id"] or f"C-LIVE-{uuid.uuid4().hex[:8].upper()}"
        record["policy_id"] = record["policy_id"] or "POL-UNKNOWN"
        record["claimant_age"] = record["claimant_age"] or 0  # stored for completeness; never a model input
        reported = pd.Timestamp(record["report_ts"])
        record["report_ts"] = reported.tz_localize(None) if reported.tzinfo else reported  # history is local wall time
        frame = derive_claim_fields(pd.DataFrame([record]))
        features = st.store.features_for(frame)
        features = features.merge(frame[["claim_id", "narrative"]], on="claim_id")
        probability = float(st.bundle.predict(features, features["narrative"])[0])
        value = float(business.expected_value(np.array([probability]), frame["total_claimed"], st.policy)[0])
        enters = value > max(st.queue_cutoff, 0.0)
        band = str(business.risk_band(np.array([probability]), np.array([enters]), st.policy)[0])
        explanation = st.bundle.explain(features, features["narrative"])[0]
        SCORED.labels(band).inc()
        PROBABILITY.observe(probability)
        result = {
            "claim_id": record["claim_id"],
            "probability": round(probability, 4),
            "expected_value": round(value, 2),
            "band": band,
            "would_enter_queue": bool(enters),
            "queue_cutoff": round(st.queue_cutoff, 2),
            "total_claimed": float(frame["total_claimed"].iloc[0]),
            "reasons_for": explanation["for"],
            "reasons_against": explanation["against"],
            "narrative_risk": explanation["narrative_risk"],
            "model_version": st.bundle.metadata.get("version"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        with st.lock:
            st.live.appendleft({**result, "loss_type": record["loss_type"], "city": record["city"],
                                "received_at": datetime.now(UTC).isoformat(timespec="seconds")})
        return result

    @app.get("/api/queue")
    def queue(limit: int = 30) -> dict[str, Any]:
        st = state()
        top = st.scored.sort_values("queue_rank").head(max(1, min(limit, 200)))
        return {"month": str(top["month"].iloc[0]) if len(top) else None, "claims": _records(top[QUEUE_FIELDS])}

    @app.get("/api/live")
    def live(limit: int = 50) -> dict[str, Any]:
        st = state()
        with st.lock:
            events = list(st.live)[: max(1, min(limit, 200))]
        return {"events": events, "count": len(st.live)}

    @app.get("/api/claims/{claim_id}")
    def claim(claim_id: str) -> dict[str, Any]:
        st = state()
        match = st.scored[st.scored["claim_id"] == claim_id]
        if match.empty:
            raise HTTPException(status_code=404, detail="Unknown claim")
        return _records(match)[0]

    @app.post("/api/copilot/ask")
    def ask(body: Question) -> dict[str, Any]:
        st = state()
        try:
            answer = st.copilot.ask(body.question, body.claim_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        COPILOT.labels(answer["mode"]).inc()
        return answer

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    site = load_settings().paths.site
    if site.exists():
        app.mount("/", StaticFiles(directory=site, html=True), name="dashboard")
    return app


app = create_app()
