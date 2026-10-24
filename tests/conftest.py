"""Shared fixtures. Nothing here touches the committed artifacts: every run works in a temp dir."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from fraudops.config import REPO_ROOT, Paths, Settings


def make_settings(work) -> Settings:
    return Settings(Paths(repo=REPO_ROOT, work=work))


@pytest.fixture(scope="session")
def small_claims(tmp_path_factory) -> pd.DataFrame:
    from fraudops.generate import generate_claims

    settings = make_settings(tmp_path_factory.mktemp("claims"))
    return generate_claims(settings, claims_per_month=90)


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory) -> Settings:
    """A complete pipeline run on a small dataset, without a local LLM."""
    os.environ["FRAUDOPS_NO_LLM"] = "1"
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from fraudops import features, generate, insights, score, train
    from fraudops.rag import knowledge

    settings = make_settings(tmp_path_factory.mktemp("pipeline"))
    generate.run(settings, claims_per_month=120)
    features.run(settings)
    train.run(settings)
    score.run(settings)
    insights.run(settings)
    knowledge.run(settings)
    return settings
