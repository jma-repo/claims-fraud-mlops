"""Command line entry point: `fraudops <step>` or `fraudops all` for the whole pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable

from fraudops.config import Settings, settings


def _generate(s: Settings, args: argparse.Namespace) -> str:
    from fraudops import generate

    per_month = 160 if getattr(args, "small", False) else None
    frame = generate.run(s, claims_per_month=per_month)
    return f"{len(frame):,} claims, fraud rate {frame['is_fraud'].mean():.1%}"


def _features(s: Settings, args: argparse.Namespace) -> str:
    from fraudops import features

    frame = features.run(s)
    return f"{frame.shape[1]} columns for {len(frame):,} claims"


def _train(s: Settings, args: argparse.Namespace) -> str:
    from fraudops import train

    summary = train.run(s)
    test = summary["champion"]["test"]
    return (f"champion={summary['champion']['candidate']} v{summary['champion']['version']} ({summary['decision']}), "
            f"test PR-AUC {test['pr_auc']:.3f}, net savings ${test['net_savings']:,.0f}/month")


def _score(s: Settings, args: argparse.Namespace) -> str:
    from fraudops import score

    result = score.run(s)
    return f"{result['claims']} claims scored, {result['queue']} in the SIU queue"


def _insights(s: Settings, args: argparse.Namespace) -> str:
    from fraudops import insights

    result = insights.run(s)
    return f"{result['segments']} segments, {result['emerging']} emerging pattern(s), {result['rings']} rings"


def _kb(s: Settings, args: argparse.Namespace) -> str:
    from fraudops.rag import knowledge

    result = knowledge.run(s)
    return f"{result['documents']} documents, {result['chunks']} chunks, retrieval={result['mode']}"


def _evaluate(s: Settings, args: argparse.Namespace) -> str:
    from fraudops.rag import evaluate

    result = evaluate.run(s, with_generation=not getattr(args, "retrieval_only", False))
    return f"hit@3={result['metrics']['hit_at_3']:.2f}, mode={result['mode']}"


def _report(s: Settings, args: argparse.Namespace) -> str:
    from fraudops.report import build

    path = build.run(s)
    return f"dashboard written to {path.relative_to(s.paths.work)}"


STEPS: dict[str, tuple[str, Callable[[Settings, argparse.Namespace], str]]] = {
    "generate": ("Generate synthetic claims", _generate),
    "features": ("Build features with DuckDB", _features),
    "train": ("Train, compare, register in MLflow", _train),
    "score": ("Score the newest month and build the SIU queue", _score),
    "insights": ("Segments, emerging patterns, rings, drift", _insights),
    "kb": ("Refresh the copilot knowledge base and index", _kb),
    "evaluate": ("Evaluate the copilot (retrieval and answers)", _evaluate),
    "report": ("Build the dashboard", _report),
}


def _run_steps(names: list[str], args: argparse.Namespace) -> int:
    s = settings()
    for position, name in enumerate(names, start=1):
        title, step = STEPS[name]
        print(f"[{position}/{len(names)}] {title} ...", flush=True)
        started = time.time()
        detail = step(s, args)
        print(f"      {detail}  ({time.time() - started:.1f}s)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudops", description="Claims fraud triage PoC pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (title, _) in STEPS.items():
        step = sub.add_parser(name, help=title)
        if name == "generate":
            step.add_argument("--small", action="store_true", help="tiny dataset for smoke tests")
        if name == "evaluate":
            step.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    everything = sub.add_parser("all", help="run every step in order")
    everything.add_argument("--small", action="store_true", help="tiny dataset for CI smoke runs")
    everything.add_argument("--no-llm", action="store_true", help="never call the local LLM (CI, laptops without one)")
    serve = sub.add_parser("serve", help="run the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    replay = sub.add_parser("replay", help="stream the newest month's claims into the API, like live FNOL events")
    replay.add_argument("--url", default="http://127.0.0.1:8000")
    replay.add_argument("--rate", type=float, default=2.0, help="claims per second")
    replay.add_argument("--limit", type=int, default=0, help="stop after N claims (0 = all)")

    args = parser.parse_args(argv)
    if getattr(args, "no_llm", False):
        os.environ["FRAUDOPS_NO_LLM"] = "1"
    if args.command == "all":
        args.retrieval_only = getattr(args, "no_llm", False)
        return _run_steps(list(STEPS), args)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("fraudops.api.app:app", host=args.host, port=args.port)
        return 0
    if args.command == "replay":
        from fraudops.replay import replay as run_replay

        return run_replay(settings(), args.url, args.rate, args.limit)
    return _run_steps([args.command], args)


if __name__ == "__main__":
    sys.exit(main())
