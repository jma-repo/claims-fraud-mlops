"""Train, compare and register: rules baseline vs logistic regression vs LightGBM.

Selection is on business value, not AUC: the candidate with the highest validation net
savings at the SIU's real capacity wins, its test result is reported once, and it only
replaces the registered champion if it beats the champion's recorded test savings.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudops import business
from fraudops.config import Settings
from fraudops.features import CATEGORICAL, MODEL_FEATURES, NUMERIC, TEXT_FEATURE
from fraudops.model import ModelBundle, fit_calibrator, model_frame
from fraudops.text_model import NarrativeModel

EXPERIMENT = "claims-fraud-triage"
RUNTIME_PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "duckdb", "pyarrow"]


def load_training_frame(settings: Settings) -> pd.DataFrame:
    claims = pd.read_parquet(settings.paths.claims,
                             columns=["claim_id", "narrative", "total_claimed", "is_fraud", "fraud_amount", "scheme"])
    features = pd.read_parquet(settings.paths.features)
    return features.merge(claims, on="claim_id", validate="one_to_one")


def split(frame: pd.DataFrame, settings: Settings) -> dict[str, pd.DataFrame]:
    cfg = settings.pipeline
    new_month = cfg["data"]["new_month"]
    train_until, valid_until = cfg["split"]["train_until"], cfg["split"]["valid_until"]
    history = frame[frame["month"] < new_month]
    return {
        "train": history[history["month"] <= train_until].reset_index(drop=True),
        "valid": history[(history["month"] > train_until) & (history["month"] <= valid_until)].reset_index(drop=True),
        "test": history[history["month"] > valid_until].reset_index(drop=True),
    }


def build_logreg() -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
         NUMERIC + [TEXT_FEATURE]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))])


def build_lightgbm(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1,
    )


def score_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(labels, probability)),
            "pr_auc": float(average_precision_score(labels, probability))}


def policy_metrics(frame: pd.DataFrame, probability: np.ndarray, policy: business.SiuPolicy) -> dict[str, float]:
    priority = business.expected_value(probability, frame["total_claimed"], policy)
    return business.monthly_outcome(frame, priority, policy, floor=0.0)


def _pr_points(labels: np.ndarray, probability: np.ndarray, n: int = 80) -> dict[str, list[float]]:
    precision, recall, _ = precision_recall_curve(labels, probability)
    keep = np.unique(np.linspace(0, len(precision) - 1, n).astype(int))
    return {"precision": precision[keep].round(4).tolist(), "recall": recall[keep].round(4).tolist()}


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items() if isinstance(value, (int, float, np.floating))}


def run(settings: Settings) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient

    started = time.time()
    paths = settings.paths
    policy = business.SiuPolicy.from_settings(settings)
    seed = settings.seed
    parts = split(load_training_frame(settings), settings)
    train, valid, test = parts["train"], parts["valid"], parts["test"]

    # Narrative model: out-of-fold scores for training rows so the tabular model never sees
    # a text score that was fitted on its own label.
    narrative = NarrativeModel(seed)
    train_risk = narrative.out_of_fold(train["narrative"], train["is_fraud"])
    narrative.fit(train["narrative"], train["is_fraud"])
    matrices = {
        "train": model_frame(train, train_risk),
        "valid": model_frame(valid, narrative.predict(valid["narrative"])),
        "test": model_frame(test, narrative.predict(test["narrative"])),
    }
    labels = {name: parts[name]["is_fraud"].to_numpy() for name in parts}

    mlflow.set_tracking_uri(paths.mlflow_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        EXPERIMENT, artifact_location=paths.mlflow_artifacts.resolve().as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)

    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}") as parent:
        mlflow.log_params({
            "rows_train": len(train), "rows_valid": len(valid), "rows_test": len(test),
            "fraud_rate_train": round(float(labels["train"].mean()), 4),
            "siu_capacity": policy.capacity, "investigation_cost": policy.investigation_cost,
            "recovery_rate": policy.recovery_rate, "seed": seed,
        })

        # Baseline: the red-flag scorecard the SIU uses today.
        with mlflow.start_run(run_name="rules-scorecard", nested=True):
            entry = {}
            for name in ("valid", "test"):
                frame = parts[name]
                priority = business.rules_priority(frame, frame["total_claimed"])
                points = business.rule_points(frame).to_numpy()
                entry[name] = {**score_metrics(labels[name], points), **business.monthly_outcome(frame, priority, policy)}
                mlflow.log_metrics(_prefixed(name, entry[name]))
            mlflow.set_tag("candidate", "rules")
            results["rules"] = entry

        for name, estimator in (("logreg", build_logreg()), ("lightgbm", build_lightgbm(seed))):
            with mlflow.start_run(run_name=name, nested=True):
                if name == "lightgbm":
                    estimator.fit(matrices["train"], labels["train"],
                                  eval_X=(matrices["valid"],), eval_y=(labels["valid"],), eval_metric="average_precision",
                                  callbacks=[lgb.early_stopping(100, verbose=False)])
                    mlflow.log_params({k: v for k, v in estimator.get_params().items()
                                       if k in ("learning_rate", "num_leaves", "min_child_samples", "colsample_bytree", "subsample")})
                    mlflow.log_param("best_iteration", estimator.best_iteration_)
                else:
                    estimator.fit(matrices["train"], labels["train"])
                    mlflow.log_param("C", estimator.named_steps["clf"].C)
                calibrator = fit_calibrator(estimator.predict_proba(matrices["valid"])[:, 1], labels["valid"])
                entry = {}
                for split_name in ("valid", "test"):
                    probability = calibrator.predict(estimator.predict_proba(matrices[split_name])[:, 1])
                    entry[split_name] = {**score_metrics(labels[split_name], probability),
                                         **policy_metrics(parts[split_name], probability, policy)}
                    entry[f"{split_name}_probability"] = probability
                    mlflow.log_metrics(_prefixed(split_name, entry[split_name]))
                mlflow.set_tag("candidate", name)
                results[name] = entry
                fitted[name] = (estimator, calibrator)

        champion_name = max(fitted, key=lambda name: results[name]["valid"]["net_savings"])
        estimator, calibrator = fitted[champion_name]
        champion_test = results[champion_name]["test"]
        bundle = ModelBundle(narrative=narrative, estimator=estimator, kind=champion_name, calibrator=calibrator)

        # Explainability summary on the test months: mean |contribution| per feature.
        contributions = bundle.contributions(matrices["test"])
        importance = contributions.abs().mean().sort_values(ascending=False)

        mlflow.log_metrics({f"champion_{k}": v for k, v in _prefixed("test", champion_test).items()})
        mlflow.log_metric("test_net_savings_per_month", champion_test["net_savings"])
        mlflow.set_tags({"champion_candidate": champion_name, "selection_metric": "valid_net_savings"})
        mlflow.log_dict({"importance": importance.round(4).to_dict()}, "explainability/importance.json")

        bundle.metadata = {"candidate": champion_name}
        # Integers become floats in the logged schema: a missing value at inference time must not break it.
        example = test[MODEL_FEATURES[:-1] + ["narrative"]].head(5).astype({column: float for column in NUMERIC})
        with tempfile.TemporaryDirectory() as staging:
            bundle.save(Path(staging))
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(paths.repo / "src" / "fraudops" / "mlflow_entry.py"),
                artifacts={"bundle": str(Path(staging) / "bundle.joblib")},
                code_paths=[str(paths.repo / "src" / "fraudops")],
                pip_requirements=[f"{package}=={version(package)}" for package in RUNTIME_PACKAGES],
                input_example=example,
                registered_model_name=settings.pipeline["model"]["name"],
            )
        model_name = settings.pipeline["model"]["name"]
        new_version = str(info.registered_model_version)

        # Champion / challenger gate.
        decision, previous = "promoted", None
        try:
            current = client.get_model_version_by_alias(model_name, "champion")
            previous = {"version": current.version,
                        "net_savings": client.get_run(current.run_id).data.metrics.get("test_net_savings_per_month")}
        except Exception:  # no champion yet
            current = None
        threshold = float(settings.pipeline["model"]["min_improvement"])
        if previous and previous["net_savings"] is not None and champion_test["net_savings"] < previous["net_savings"] + threshold:
            decision = "kept_previous"
        client.set_model_version_tag(model_name, new_version, "decision", decision)
        client.set_model_version_tag(model_name, new_version, "candidate", champion_name)
        if decision == "promoted":
            client.set_registered_model_alias(model_name, "champion", new_version)
        else:
            client.set_registered_model_alias(model_name, "challenger", new_version)
        mlflow.set_tag("decision", decision)
        run_id = parent.info.run_id

    # Business view of the test months for the dashboard.
    test_priorities = {
        "model": business.expected_value(results[champion_name]["test_probability"], test["total_claimed"], policy),
        "rules": business.rules_priority(test, test["total_claimed"]),
    }
    capacities = list(range(0, 125, 5))
    curve = business.capacity_curve(test, test_priorities, policy, capacities)

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    metadata = {
        "model_name": model_name,
        "version": new_version if decision == "promoted" else (previous or {}).get("version"),
        "candidate": champion_name,
        "mlflow_run_id": run_id,
        "trained_at": trained_at,
        "train_months": [str(train["month"].min()), str(train["month"].max())],
        "features": MODEL_FEATURES,
        "best_iteration": int(getattr(estimator, "best_iteration_", 0) or 0),
        "test": {k: round(float(v), 4) for k, v in champion_test.items()},
    }
    if decision == "promoted":
        bundle.metadata = metadata
        bundle.save(paths.model_dir)

    summary = {
        "trained_at": trained_at,
        "decision": decision,
        "previous_champion": previous,
        "champion": metadata,
        "policy": policy.__dict__,
        "data": {name: {"rows": len(parts[name]), "fraud_rate": round(float(labels[name].mean()), 4),
                        "months": [str(parts[name]["month"].min()), str(parts[name]["month"].max())]}
                 for name in parts},
        "candidates": {name: {split_name: {k: round(float(v), 4) for k, v in entry[split_name].items()}
                              for split_name in ("valid", "test")}
                       for name, entry in results.items()},
        "random_test": {k: round(float(v), 2) for k, v in business.random_outcome(test, policy).items()},
        "capacity_curve": {k: [round(float(x), 2) for x in v] for k, v in curve.items()},
        "pr_curve": {name: _pr_points(labels["test"], results[name]["test_probability"]) for name in fitted},
        "importance": importance.head(15).round(4).to_dict(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
