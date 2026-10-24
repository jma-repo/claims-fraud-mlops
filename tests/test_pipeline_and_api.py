"""End-to-end: a small pipeline run, the MLflow registry it leaves behind, and the API on top of it."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudops.api.app import build_state, create_app
from fraudops.replay import to_payload


def test_pipeline_writes_every_artifact_the_api_and_dashboard_need(pipeline):
    paths = pipeline.paths
    for path in [paths.model_dir / "bundle.joblib", paths.model_dir / "metadata.json", paths.history, paths.scored,
                 paths.replay, paths.kb_index, paths.reports / "metrics.json", paths.reports / "segments.json",
                 paths.reports / "rings.json", paths.reports / "drift.json", paths.kb_generated / "model-card.md"]:
        assert path.exists(), path


def test_training_compares_candidates_and_registers_a_champion(pipeline):
    import mlflow
    from mlflow import MlflowClient

    metrics = json.loads((pipeline.paths.reports / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["candidates"]) == {"rules", "logreg", "lightgbm"}
    assert metrics["decision"] == "promoted"
    mlflow.set_tracking_uri(pipeline.paths.mlflow_uri)
    champion = MlflowClient().get_model_version_by_alias(pipeline.pipeline["model"]["name"], "champion")
    assert str(champion.version) == str(metrics["champion"]["version"])


def test_queue_respects_capacity_and_ranks_by_expected_value(pipeline):
    scored = pd.read_parquet(pipeline.paths.scored)
    queue = scored[scored["in_queue"]]
    assert 0 < len(queue) <= pipeline.business["siu"]["capacity_per_month"]
    assert (queue["expected_value"] > 0).all()
    assert queue["expected_value"].min() >= scored.loc[~scored["in_queue"], "expected_value"].max()
    assert scored["queue_rank"].is_unique


@pytest.fixture(scope="module")
def client(pipeline):
    app = create_app(state_factory=lambda: build_state(pipeline))
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_served_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model"]["candidate"] in {"lightgbm", "logreg"}


def test_online_scores_match_the_batch_scores(pipeline, client):
    """The same claim scored through the API and through the batch job gets the same probability."""
    replay = pd.read_parquet(pipeline.paths.replay).sort_values("report_ts").head(10)
    batch = pd.read_parquet(pipeline.paths.scored).set_index("claim_id")
    for record in replay.to_dict(orient="records"):
        response = client.post("/api/score", json=to_payload(record))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["probability"] == pytest.approx(batch.loc[body["claim_id"], "probability"], abs=1e-4)
        assert body["band"] in {"Investigate", "Review", "Standard", "Fast-track"}
        assert body["reasons_for"]


def test_invalid_claims_are_rejected(pipeline, client):
    record = to_payload(pd.read_parquet(pipeline.paths.replay).iloc[0].to_dict())
    assert client.post("/api/score", json={**record, "loss_type": "meteor"}).status_code == 422
    assert client.post("/api/score", json={**record, "n_injured": 9, "n_occupants": 1}).status_code == 422
    assert client.post("/api/score", json={**record, "unexpected": 1}).status_code == 422


def test_queue_claim_and_copilot_endpoints(client):
    queue = client.get("/api/queue?limit=5").json()["claims"]
    assert [c["queue_rank"] for c in queue] == sorted(c["queue_rank"] for c in queue)
    assert client.get(f"/api/claims/{queue[0]['claim_id']}").status_code == 200
    assert client.get("/api/claims/C-NOPE").status_code == 404
    answer = client.post("/api/copilot/ask", json={"question": "What should be in an SIU referral?"}).json()
    assert answer["mode"] in {"extractive", "abstained"} and answer["local_only"] is True
    claim_answer = client.post("/api/copilot/ask", json={"question": "Why is this claim flagged?", "claim_id": queue[0]["claim_id"]})
    assert claim_answer.status_code == 200 and claim_answer.json()["sources"][0]["label"] == "C"


def test_metrics_endpoint_exposes_service_and_drift_metrics(client):
    text = client.get("/metrics").text
    for name in ["fraudops_requests_total", "fraudops_scored_claims_total", "fraudops_feature_psi", "fraudops_model_info"]:
        assert name in text


def test_responses_carry_security_headers(client):
    headers = client.get("/api/health").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
