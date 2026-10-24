"""Paths and settings shared by every pipeline step and the API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(os.environ.get("FRAUDOPS_REPO", Path(__file__).resolve().parents[2])).resolve()


@dataclass(frozen=True)
class Paths:
    """Source files live in the repo; generated files live in the work directory.

    The two are the same folder by default. Tests point FRAUDOPS_HOME at a temp folder so a
    test run never overwrites the committed artifacts.
    """

    repo: Path
    work: Path

    @classmethod
    def default(cls) -> Paths:
        work = Path(os.environ.get("FRAUDOPS_HOME", REPO_ROOT)).resolve()
        return cls(repo=REPO_ROOT, work=work)

    # Sources
    @property
    def configs(self) -> Path:
        return self.repo / "configs"

    @property
    def sql(self) -> Path:
        return self.repo / "sql"

    @property
    def knowledge(self) -> Path:
        return self.repo / "knowledge"

    @property
    def templates(self) -> Path:
        return self.repo / "src" / "fraudops" / "report"

    # Regenerable working data (git-ignored)
    @property
    def data(self) -> Path:
        return self.work / "data"

    @property
    def claims(self) -> Path:
        return self.data / "claims.parquet"

    @property
    def features(self) -> Path:
        return self.data / "features.parquet"

    @property
    def mlflow_uri(self) -> str:
        return "sqlite:///" + (self.work / "mlflow.db").as_posix()

    @property
    def mlflow_artifacts(self) -> Path:
        return self.work / "mlartifacts"

    # Committed snapshot: everything the API and the dashboard need, so a fresh clone runs
    # without retraining.
    @property
    def artifacts(self) -> Path:
        return self.work / "artifacts"

    @property
    def model_dir(self) -> Path:
        return self.artifacts / "model"

    @property
    def serving(self) -> Path:
        return self.artifacts / "serving"

    @property
    def history(self) -> Path:
        return self.serving / "history.parquet"

    @property
    def scored(self) -> Path:
        return self.serving / "scored_latest.parquet"

    @property
    def replay(self) -> Path:
        return self.serving / "replay_claims.parquet"

    @property
    def kb_dir(self) -> Path:
        return self.artifacts / "kb"

    @property
    def kb_index(self) -> Path:
        return self.kb_dir / "index.parquet"

    @property
    def kb_generated(self) -> Path:
        return self.kb_dir / "generated"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    @property
    def site(self) -> Path:
        return self.work / "site"


@dataclass(frozen=True)
class Settings:
    paths: Paths

    @cached_property
    def pipeline(self) -> dict[str, Any]:
        return _load_yaml(self.paths.configs / "pipeline.yaml")

    @cached_property
    def business(self) -> dict[str, Any]:
        return _load_yaml(self.paths.configs / "business.yaml")

    @property
    def seed(self) -> int:
        return int(self.pipeline["seed"])


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def settings() -> Settings:
    return Settings(Paths.default())
