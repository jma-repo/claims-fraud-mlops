"""Copilot evaluation, tracked in MLflow next to the fraud model (the LangSmith role in this PoC).

Retrieval is scored on every run (hit@k, MRR); answers are scored when a local model is available
(expected terms present, citations present, out-of-scope questions declined, latency).
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any

import yaml

from fraudops.config import Settings
from fraudops.rag.copilot import NOT_COVERED, ClaimDirectory, Copilot
from fraudops.rag.llm import LocalLLM, llm_disabled
from fraudops.rag.retrieval import CHUNK_CHARS, LEXICAL_WEIGHT, SEMANTIC_WEIGHT, KnowledgeBase

EXPERIMENT = "copilot-rag-eval"
CLAIM_QUESTION = "Why is this claim flagged, and what should the investigator check first?"


def _declined(result: dict[str, Any]) -> bool:
    return result["mode"] == "abstained" or NOT_COVERED.lower().rstrip(".") in result["answer"].lower()


def build_copilot(settings: Settings, generate: bool) -> tuple[Copilot, LocalLLM | None]:
    paths = settings.paths
    llm = None if llm_disabled() else LocalLLM()
    if llm is not None and not llm.status()["ready"]:
        llm = None
    claims = ClaimDirectory.from_artifacts(paths.scored, paths.reports)
    return Copilot(KnowledgeBase.load(paths.kb_index), llm, claims, generate=generate), llm


def run(settings: Settings, with_generation: bool = True) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    paths = settings.paths
    cases = yaml.safe_load((paths.repo / "eval" / "copilot_eval.yaml").read_text(encoding="utf-8"))["questions"]
    copilot, llm = build_copilot(settings, generate=with_generation)
    generating = with_generation and llm is not None

    scored = copilot.claims.scored
    top_claim = scored.sort_values("queue_rank").iloc[0]["claim_id"]
    segments = json.loads((paths.reports / "segments.json").read_text(encoding="utf-8"))
    claim_cases = [{"id": "claim-top-queue", "question": CLAIM_QUESTION, "claim_id": top_claim}]
    if segments["emerging"]:
        members = scored[scored["claim_id"].isin(segments["emerging"][0]["claim_ids"])]
        claim_cases.append({"id": "claim-emerging", "question": CLAIM_QUESTION,
                            "claim_id": members.sort_values("total_claimed").iloc[-1]["claim_id"],
                            "expect_terms": ["storage"]})

    rows, results = [], {}
    for case in cases + claim_cases:
        result = copilot.ask(case["question"], case.get("claim_id"))
        results[case["id"]] = result
        kb_hits = [s for s in result["sources"] if s["label"] != "C"]
        kb_sources = [s["source"] for s in kb_hits]
        expected = case.get("expect_sources", [])
        rank = next((i for i, source in enumerate(kb_sources, start=1) if source in expected), None)
        answer = result["answer"].lower()
        terms = case.get("expect_terms", [])
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "claim_id": case.get("claim_id"),
            "out_of_scope": bool(case.get("out_of_scope")),
            "expected_sources": expected,
            "retrieved": kb_sources,
            "top_score": kb_hits[0]["score"] if kb_hits else None,
            "retrieval": result["retrieval"],
            "rank": rank,
            "declined": _declined(result),
            "term_recall": (sum(term.lower() in answer for term in terms) / len(terms)) if terms and generating else None,
            "cited": "[" in result["answer"],
            "mode": result["mode"],
            "elapsed_ms": result["elapsed_ms"],
            "answer": result["answer"],
        })

    in_scope = [r for r in rows if not r["out_of_scope"] and r["expected_sources"]]
    out_scope = [r for r in rows if r["out_of_scope"]]
    metrics = {
        "hit_at_1": sum(r["rank"] == 1 for r in in_scope) / len(in_scope),
        "hit_at_3": sum(r["rank"] is not None and r["rank"] <= 3 for r in in_scope) / len(in_scope),
        "mrr": sum(1 / r["rank"] for r in in_scope if r["rank"]) / len(in_scope),
        "out_of_scope_declined": sum(r["declined"] for r in out_scope) / len(out_scope),
        "in_scope_wrongly_declined": sum(r["declined"] for r in in_scope) / len(in_scope),
    }
    if generating:
        answered = [r for r in rows if not r["out_of_scope"]]
        recalls = [r["term_recall"] for r in answered if r["term_recall"] is not None]
        latencies = sorted(r["elapsed_ms"] for r in answered)
        metrics.update({
            "term_recall": statistics.mean(recalls) if recalls else 0.0,
            "citation_rate": sum(r["cited"] for r in answered if not r["declined"]) / max(1, sum(not r["declined"] for r in answered)),
            "latency_p50_ms": latencies[len(latencies) // 2],
            "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        })
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    mode = f"{'generation' if generating else 'retrieval-only'} / {rows[0]['retrieval']} retrieval"

    mlflow.set_tracking_uri(paths.mlflow_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=paths.mlflow_artifacts.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="generation" if generating else "retrieval-only"):
        mlflow.log_params({"chunk_chars": CHUNK_CHARS, "semantic_weight": SEMANTIC_WEIGHT, "lexical_weight": LEXICAL_WEIGHT,
                           "top_k": copilot.k, "chunks": len(copilot.kb.chunks),
                           "embed_model": copilot.kb.meta.get("embed_model") or "none",
                           "chat_model": (llm.choose().chat if generating else "none"), "questions": len(rows)})
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"rows": rows}, "rag_eval_rows.json")

    report = {"mode": mode, "metrics": metrics, "rows": rows}
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "rag_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if generating:
        # Recorded answers for the static dashboard, which has no model server behind it.
        picks = ("claim-top-queue", "claim-emerging", "release-fee", "oos-deductible")
        examples = [{**results[name], "recorded": True} for name in picks if name in results]
        (paths.reports / "copilot_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return report
