"""The deployable model bundle: narrative model + tabular model + calibration + reason codes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraudops.features import CATEGORICAL, LOSS_TYPES, MODEL_FEATURES, NUMERIC, PROVINCES, TEXT_FEATURE
from fraudops.text_model import NarrativeModel

CATEGORIES = {"loss_type": LOSS_TYPES, "province": PROVINCES}


def model_frame(features: pd.DataFrame, narrative_risk: np.ndarray) -> pd.DataFrame:
    """The exact matrix both training and serving feed to the tabular model."""
    frame = features[NUMERIC].astype(float).copy()
    for column in CATEGORICAL:
        frame[column] = pd.Categorical(features[column], categories=CATEGORIES[column])
    frame[TEXT_FEATURE] = np.asarray(narrative_risk, dtype=float)
    return frame[MODEL_FEATURES]


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _count(value: float, one: str, many: str) -> str:
    n = int(round(value))
    return f"{n} {one if n == 1 else many}"


# Reason codes an investigator can read. Each maps a feature value to plain language.
REASON_TEXT: dict[str, Any] = {
    "phone_prior_claims_365d": lambda r: f"Phone number appears on {_count(r['phone_prior_claims_365d'], 'other claim', 'other claims')} in the past year",
    "address_prior_claims_365d": lambda r: f"Address appears on {_count(r['address_prior_claims_365d'], 'other claim', 'other claims')} in the past year",
    "rep_clinic_prior_claims_365d": lambda r: f"Same representative and clinic paired on {_count(r['rep_clinic_prior_claims_365d'], 'prior claim', 'prior claims')}",
    "clinic_prior_claims_90d": lambda r: f"Clinic on {_count(r['clinic_prior_claims_90d'], 'injury claim', 'injury claims')} in the last 90 days",
    "shop_prior_claims_90d": lambda r: f"Body shop on {_count(r['shop_prior_claims_90d'], 'claim', 'claims')} in the last 90 days",
    "shop_avg_repair_ratio_180d": lambda r: f"Body shop's recent estimates average {r['shop_avg_repair_ratio_180d']:.0%} of vehicle value",
    "tow_prior_claims_90d": lambda r: f"Tow operator on {_count(r['tow_prior_claims_90d'], 'claim', 'claims')} in the last 90 days",
    "tow_avg_storage_days_180d": lambda r: f"Tow operator's vehicles sit in storage {r['tow_avg_storage_days_180d']:.0f} days on average",
    "policy_tenure_days": lambda r: f"Policy was {_count(r['policy_tenure_days'], 'day', 'days')} old at the time of loss",
    "report_delay_days": lambda r: f"Reported {_count(r['report_delay_days'], 'day', 'days')} after the loss",
    "n_injured": lambda r: f"{int(r['n_injured'])} of {int(r['n_occupants'])} occupants claim injury",
    "injured_share": lambda r: f"{int(r['n_injured'])} of {int(r['n_occupants'])} occupants claim injury",
    "n_occupants": lambda r: f"{_count(r['n_occupants'], 'occupant', 'occupants')} in the vehicle",
    "soft_tissue_only": lambda r: "Soft-tissue injuries only" if r["soft_tissue_only"] else "Injuries are not soft-tissue only",
    "night_loss": lambda r: f"Loss at {int(r['loss_hour']):02d}:00",
    "loss_hour": lambda r: f"Loss at {int(r['loss_hour']):02d}:00",
    "police_report": lambda r: "No police attendance" if not r["police_report"] else "Police report on file",
    "witnesses": lambda r: "No independent witnesses" if not r["witnesses"] else _count(r["witnesses"], "independent witness", "independent witnesses"),
    "represented": lambda r: "Claimant represented by a paralegal or lawyer" if r["represented"] else "Claimant not represented",
    "narrative_risk": lambda r: "Adjuster notes read like confirmed fraud files",
    "log_total_claimed": lambda r: f"Total claimed {_money(math.expm1(r['log_total_claimed']))}",
    "log_injury_reserve": lambda r: f"Injury reserve {_money(math.expm1(r['log_injury_reserve']))}",
    "log_repair_estimate": lambda r: f"Repair estimate {_money(math.expm1(r['log_repair_estimate']))}",
    "repair_to_value": lambda r: f"Repair estimate is {r['repair_to_value']:.0%} of vehicle value",
    "vehicle_value": lambda r: f"Vehicle value {_money(r['vehicle_value'])}",
    "vehicle_age": lambda r: f"Vehicle is {_count(r['vehicle_age'], 'year', 'years')} old",
    "total_loss": lambda r: "Declared a total loss" if r["total_loss"] else "Repairable damage",
    "tow_distance_km": lambda r: f"Towed {r['tow_distance_km']:.0f} km",
    "storage_days": lambda r: f"{_count(r['storage_days'], 'day', 'days')} in storage",
    "tow_fee": lambda r: f"Towing and storage fees {_money(r['tow_fee'])}",
    "claimant_age": lambda r: f"Claimant is {int(r['claimant_age'])}",
    "prior_claims_3y": lambda r: f"{_count(r['prior_claims_3y'], 'prior claim', 'prior claims')} in 3 years",
    "loss_type": lambda r: f"Loss type: {str(r['loss_type']).replace('_', ' ')}",
    "province": lambda r: f"Province: {r['province']}",
}


def describe(feature: str, row: pd.Series) -> str:
    try:
        return REASON_TEXT[feature](row)
    except (KeyError, TypeError, ValueError):
        return feature.replace("_", " ")


@dataclass
class ModelBundle:
    narrative: NarrativeModel
    estimator: Any
    kind: str                                   # "lightgbm" or "logreg"
    calibrator: PlattCalibrator
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- scoring --------------------------------------------------------------------------
    def narrative_risk(self, narratives: pd.Series | list[str]) -> np.ndarray:
        return self.narrative.predict(narratives)

    def raw_score(self, matrix: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(matrix)[:, 1]

    def predict(self, features: pd.DataFrame, narratives: pd.Series | list[str]) -> np.ndarray:
        matrix = model_frame(features, self.narrative_risk(narratives))
        return self.calibrator.predict(self.raw_score(matrix))

    # --- explanations ---------------------------------------------------------------------
    def contributions(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the log-odds of fraud (SHAP for trees, coef x value for logreg)."""
        if self.kind == "lightgbm":
            values = self.estimator.predict(matrix, pred_contrib=True)[:, :-1]
            return pd.DataFrame(values, columns=MODEL_FEATURES, index=matrix.index)
        preprocess = self.estimator.named_steps["prep"]
        coefficients = self.estimator.named_steps["clf"].coef_[0]
        transformed = preprocess.transform(matrix)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        names = preprocess.get_feature_names_out()
        raw = pd.DataFrame(transformed * coefficients, columns=names, index=matrix.index)
        grouped = {feature: raw[[n for n in names if n.split("__", 1)[1] == feature or n.split("__", 1)[1].startswith(feature + "_")]].sum(axis=1)
                   for feature in MODEL_FEATURES}
        return pd.DataFrame(grouped)

    def explain(self, features: pd.DataFrame, narratives: pd.Series | list[str], top: int = 4,
                against: int = 2) -> list[dict[str, Any]]:
        narratives = pd.Series(list(narratives), index=features.index)
        risk = self.narrative_risk(narratives)
        matrix = model_frame(features, risk)
        contributions = self.contributions(matrix)
        explanations = []
        for position, (index, row) in enumerate(features.iterrows()):
            ranked = contributions.loc[index].sort_values()
            for_fraud, mitigating, seen = [], [], set()
            for feature, value in ranked[::-1].items():
                if value <= 0 or len(for_fraud) == top:
                    break
                text = describe(feature, row)
                if feature == TEXT_FEATURE:
                    terms = self.narrative.top_terms(narratives.loc[index])
                    if terms:
                        text += ": " + ", ".join(f"'{term}'" for term in terms)
                if text not in seen:
                    seen.add(text)
                    for_fraud.append({"feature": feature, "text": text, "weight": round(float(value), 3)})
            for feature, value in ranked.items():
                # Tiny negative contributions are noise to a reader, not a mitigating fact.
                if value > -0.2 or len(mitigating) == against:
                    break
                text = describe(feature, row)
                if text not in seen:
                    seen.add(text)
                    mitigating.append({"feature": feature, "text": text, "weight": round(float(value), 3)})
            explanations.append({"for": for_fraud, "against": mitigating, "narrative_risk": round(float(risk[position]), 3)})
        return explanations

    # --- persistence ----------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "bundle.joblib", compress=3)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def load(directory: Path) -> ModelBundle:
        # The bundle is produced by this repo's own pipeline; never load one from an untrusted source.
        return joblib.load(directory / "bundle.joblib")


class PlattCalibrator:
    """Sigmoid on the model's log-odds. Smooth and order-preserving: with a few dozen frauds in
    the validation months, isotonic calibration collapses scores into a handful of steps, which
    ties the queue and makes drift statistics on the score meaningless."""

    def fit(self, raw: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        model = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(raw).reshape(-1, 1), labels)
        self.slope, self.intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.slope * _logit(raw) + self.intercept)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(raw: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    return PlattCalibrator().fit(raw, labels)
