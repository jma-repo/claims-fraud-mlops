"""Feature engineering on DuckDB, shared by batch training and the online scoring API."""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pandas as pd

from fraudops.config import Settings

# The claim contract: what a first-notice-of-loss record must contain to be scored.
CLAIM_COLUMNS = [
    "claim_id", "month", "report_ts", "province", "city", "loss_type", "loss_hour", "report_delay_days",
    "policy_tenure_days", "policy_id", "vehicle_value", "vehicle_age", "n_occupants", "n_injured",
    "soft_tissue_only", "injury_reserve", "police_report", "witnesses", "claimant_age", "prior_claims_3y",
    "body_shop_id", "clinic_id", "rep_id", "tow_company_id", "tow_distance_km", "storage_days", "tow_fee",
    "repair_estimate", "phone_id", "address_id", "loss_ts", "total_loss", "total_claimed", "narrative",
]
LOSS_TYPES = ["rear_end", "intersection", "single_vehicle", "sideswipe", "parking_lot",
              "theft", "hail", "glass", "vandalism", "animal"]
PROVINCES = ["ON", "AB", "NS", "NB"]

CATEGORICAL = ["loss_type", "province"]
NUMERIC = [
    "log_total_claimed", "log_injury_reserve", "log_repair_estimate", "repair_to_value", "vehicle_value",
    "vehicle_age", "total_loss", "loss_hour", "night_loss", "n_occupants", "n_injured", "injured_share",
    "soft_tissue_only", "police_report", "witnesses", "represented", "tow_distance_km", "storage_days",
    "tow_fee", "policy_tenure_days", "report_delay_days", "prior_claims_3y",
    "phone_prior_claims_365d", "address_prior_claims_365d", "shop_prior_claims_90d",
    "shop_avg_repair_ratio_180d", "clinic_prior_claims_90d", "rep_clinic_prior_claims_365d",
    "tow_prior_claims_90d", "tow_avg_storage_days_180d",
]
# Deliberately not model inputs, even though they are in the data and some are predictive.
EXCLUDED_FEATURES = {
    "claimant_age": "protected ground under human-rights law; fraud triage must not rank people by age",
    "city": "location is a proxy for neighbourhood and, through it, for protected characteristics",
    "narrative place names": "scrubbed from adjuster notes before the text model sees them, for the same reason",
    "provider IDs": "used only through behaviour (volumes, ratios, pairings), never as an identity the model memorises",
}
TEXT_FEATURE = "narrative_risk"
MODEL_FEATURES = NUMERIC + CATEGORICAL + [TEXT_FEATURE]

LOOKBACK_DAYS = 366  # longest window in features.sql is 365 days


class DataContractError(ValueError):
    """Raised when incoming claims break the contract; the pipeline stops instead of training on bad data."""


def derive_claim_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Fields computed from a first-notice record. Shared by the generator and the API, so a
    claim typed into the API is summed exactly the way the training data was."""
    frame = frame.copy()
    frame["report_ts"] = pd.to_datetime(frame["report_ts"])
    frame["month"] = frame["report_ts"].dt.strftime("%Y-%m")
    frame["loss_ts"] = (frame["report_ts"] - pd.to_timedelta(frame["report_delay_days"], unit="D")).dt.floor("D") \
        + pd.to_timedelta(frame["loss_hour"], unit="h")
    frame["total_loss"] = (frame["loss_type"] == "theft") | (frame["repair_estimate"] >= 0.8 * frame["vehicle_value"])
    physical = frame["vehicle_value"].where(frame["total_loss"], frame["repair_estimate"])
    frame["total_claimed"] = (physical + frame["injury_reserve"] + frame["tow_fee"]).round(2)
    return frame


def validate_claims(frame: pd.DataFrame) -> None:
    missing = [column for column in CLAIM_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"missing columns: {missing}")
    problems = []
    if frame["claim_id"].duplicated().any():
        problems.append("duplicate claim_id")
    for column in ["claim_id", "report_ts", "loss_type", "province", "phone_id", "address_id"]:
        if frame[column].isna().any():
            problems.append(f"null {column}")
    for column in ["vehicle_value", "injury_reserve", "repair_estimate", "tow_fee", "total_claimed",
                   "tow_distance_km", "storage_days", "report_delay_days", "policy_tenure_days"]:
        if (frame[column] < 0).any():
            problems.append(f"negative {column}")
    if (frame["n_injured"] > frame["n_occupants"]).any():
        problems.append("n_injured > n_occupants")
    if not frame["loss_hour"].between(0, 23).all():
        problems.append("loss_hour outside 0-23")
    unknown = set(frame["loss_type"]) - set(LOSS_TYPES)
    if unknown:
        problems.append(f"unknown loss_type {sorted(unknown)}")
    unknown = set(frame["province"]) - set(PROVINCES)
    if unknown:
        problems.append(f"unknown province {sorted(unknown)}")
    if problems:
        raise DataContractError("; ".join(problems))


def feature_sql(settings: Settings) -> str:
    return (settings.paths.sql / "features.sql").read_text(encoding="utf-8")


def compute_features(claims: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Run the feature query over a frame of claims (batch path)."""
    con = duckdb.connect()
    try:
        con.register("claims", claims[CLAIM_COLUMNS])
        return con.execute(sql).df()
    finally:
        con.close()


class OnlineFeatureStore:
    """Online path: keeps claim history in an in-process DuckDB table and runs the same SQL.

    Each scored claim is appended, so the next claim sees it in its look-back windows,
    exactly as the batch query would. In production this is Redis or a managed feature store.
    """

    def __init__(self, history: pd.DataFrame, sql: str):
        self._sql = sql
        self._lock = threading.Lock()
        self._con = duckdb.connect()
        self._con.register("seed", history[CLAIM_COLUMNS])
        self._con.execute("CREATE TABLE history AS SELECT * FROM seed")
        self._con.unregister("seed")

    @classmethod
    def from_parquet(cls, path: Path, sql: str) -> OnlineFeatureStore:
        return cls(pd.read_parquet(path), sql)

    def size(self) -> int:
        with self._lock:
            return int(self._con.execute("SELECT count(*) FROM history").fetchone()[0])

    def features_for(self, claim: pd.DataFrame, append: bool = True) -> pd.DataFrame:
        """Features for one or more new claims, computed against prior history only."""
        incoming = claim[CLAIM_COLUMNS]
        # Views cannot hold bound parameters; the literal is built from a Timestamp, never from text.
        start = (pd.Timestamp(incoming["report_ts"].min()) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self._con.cursor()
            try:
                cursor.register("incoming", incoming)
                # Every window in features.sql is partitioned by one of these entities, so history
                # rows that share none of them cannot change the result. Filtering them out keeps
                # online scoring in milliseconds without touching the feature definitions.
                cursor.execute(
                    f"""
                    CREATE OR REPLACE TEMP VIEW claims AS
                    SELECT * FROM history
                    WHERE report_ts >= TIMESTAMP '{start}'
                      AND claim_id NOT IN (SELECT claim_id FROM incoming)
                      AND (phone_id IN (SELECT phone_id FROM incoming)
                           OR address_id IN (SELECT address_id FROM incoming)
                           OR body_shop_id IN (SELECT body_shop_id FROM incoming)
                           OR clinic_id IN (SELECT clinic_id FROM incoming)
                           OR tow_company_id IN (SELECT tow_company_id FROM incoming))
                    UNION ALL BY NAME
                    SELECT * FROM incoming
                    """
                )
                features = cursor.execute(self._sql).df()
                features = features[features["claim_id"].isin(incoming["claim_id"])].reset_index(drop=True)
                if append:
                    cursor.execute("DELETE FROM history WHERE claim_id IN (SELECT claim_id FROM incoming)")
                    cursor.execute("INSERT INTO history SELECT * FROM incoming")
                return features
            finally:
                cursor.close()


def run(settings: Settings) -> pd.DataFrame:
    claims = pd.read_parquet(settings.paths.claims)
    validate_claims(claims)
    features = compute_features(claims, feature_sql(settings))
    features.to_parquet(settings.paths.features, index=False)
    return features
