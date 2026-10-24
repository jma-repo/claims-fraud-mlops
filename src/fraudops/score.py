"""Score the newest month: SIU queue, risk bands and reason codes for every claim."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from fraudops import business
from fraudops.config import Settings
from fraudops.features import CLAIM_COLUMNS
from fraudops.generate import TRUTH_COLUMNS
from fraudops.model import ModelBundle

# Claim fields the dashboard and copilot need that are not features (report_ts already is one).
CONTEXT_COLUMNS = ["claim_id", "city", "narrative", "total_claimed", "body_shop_id", "clinic_id",
                   "rep_id", "tow_company_id", "phone_id", "address_id"]


def score_frame(bundle: ModelBundle, features: pd.DataFrame, context: pd.DataFrame,
                policy: business.SiuPolicy) -> pd.DataFrame:
    """Probability, expected value, queue and reasons for a batch of claims (one month)."""
    frame = features.merge(context, on="claim_id", validate="one_to_one").reset_index(drop=True)
    probability = bundle.predict(frame, frame["narrative"])
    value = business.expected_value(probability, frame["total_claimed"], policy)
    in_queue = business.select(value, policy.capacity, floor=0.0)
    explanations = bundle.explain(frame, frame["narrative"])
    order = np.argsort(-value, kind="stable")
    rank = np.empty(len(frame), dtype=int)
    rank[order] = np.arange(1, len(frame) + 1)
    return frame.assign(
        probability=probability.round(4),
        expected_value=value.round(2),
        in_queue=in_queue,
        queue_rank=rank,
        band=business.risk_band(probability, in_queue, policy),
        narrative_risk=[item["narrative_risk"] for item in explanations],
        reasons_for=[json.dumps(item["for"]) for item in explanations],
        reasons_against=[json.dumps(item["against"]) for item in explanations],
    ).sort_values("queue_rank").reset_index(drop=True)


def hidden_truth_review(scored: pd.DataFrame, truth: pd.DataFrame, policy: business.SiuPolicy) -> dict[str, Any]:
    """What the queue would have caught, using labels a real carrier would not have yet.

    Possible only because the data is synthetic; it is what makes the "emerging pattern"
    story checkable instead of anecdotal.
    """
    frame = scored.merge(truth, on="claim_id", validate="one_to_one")
    rules_chosen = business.select(business.rules_priority(frame, frame["total_claimed"]), policy.capacity)
    by_scheme = (frame[frame["is_fraud"] == 1]
                 .groupby("scheme")
                 .agg(claims=("claim_id", "size"), in_queue=("in_queue", "sum"), mean_probability=("probability", "mean"),
                      fraud_dollars=("fraud_amount", "sum"))
                 .round(3).reset_index().to_dict(orient="records"))
    fast = frame["band"] == "Fast-track"
    return {
        "model_queue": business.outcome(frame, frame["in_queue"].to_numpy(), policy),
        "rules_queue": business.outcome(frame, rules_chosen, policy),
        "by_scheme": by_scheme,
        "fast_track": {
            "share_of_claims": round(float(fast.mean()), 4),
            "fraud_dollars_missed": round(float(frame.loc[fast, "fraud_amount"].sum()), 2),
            "fraud_claims_missed": int((fast & (frame["is_fraud"] == 1)).sum()),
        },
    }


def run(settings: Settings) -> dict[str, Any]:
    paths = settings.paths
    policy = business.SiuPolicy.from_settings(settings)
    new_month = settings.pipeline["data"]["new_month"]
    bundle = ModelBundle.load(paths.model_dir)
    claims = pd.read_parquet(paths.claims)
    features = pd.read_parquet(paths.features)

    latest_features = features[features["month"] == new_month]
    latest_claims = claims[claims["month"] == new_month]
    scored = score_frame(bundle, latest_features, latest_claims[CONTEXT_COLUMNS], policy)

    paths.serving.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(paths.scored, index=False)
    # The online feature store starts from everything before the newest month; replaying the
    # newest month through the API then reproduces the batch features exactly.
    claims.loc[claims["month"] < new_month, CLAIM_COLUMNS].to_parquet(paths.history, index=False)
    latest_claims.sort_values("report_ts")[CLAIM_COLUMNS].to_parquet(paths.replay, index=False)

    review = hidden_truth_review(scored, latest_claims[["claim_id", *TRUTH_COLUMNS]], policy)
    summary = {
        "month": new_month,
        "claims": int(len(scored)),
        "queue": int(scored["in_queue"].sum()),
        "bands": scored["band"].value_counts().to_dict(),
        "queue_value": round(float(scored.loc[scored["in_queue"], "expected_value"].sum()), 2),
        "model_version": bundle.metadata.get("version"),
        "hidden_truth": review,
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "latest_month.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary
