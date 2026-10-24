"""Turning fraud scores into SIU decisions and dollars.

The model is judged on what the business can actually do with it: the SIU opens a fixed
number of files a month, each costs money, and only confirmed fraud saves money. So the
queue is ranked by expected net saving (probability x amount x recovery - cost), not by raw
probability, and every model is compared on net savings at that fixed capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudops.config import Settings

COLLISIONS = ["rear_end", "intersection", "single_vehicle", "sideswipe", "animal"]


@dataclass(frozen=True)
class SiuPolicy:
    capacity: int
    investigation_cost: float
    recovery_rate: float
    fast_track_below: float
    review_above: float

    @classmethod
    def from_settings(cls, settings: Settings) -> SiuPolicy:
        cfg = settings.business
        return cls(
            capacity=int(cfg["siu"]["capacity_per_month"]),
            investigation_cost=float(cfg["siu"]["investigation_cost"]),
            recovery_rate=float(cfg["siu"]["recovery_rate"]),
            fast_track_below=float(cfg["triage"]["fast_track_below"]),
            review_above=float(cfg["triage"]["review_above"]),
        )


def expected_value(probability: np.ndarray | pd.Series, claimed: np.ndarray | pd.Series, policy: SiuPolicy) -> np.ndarray:
    """Expected net saving of opening an investigation on a claim."""
    return np.asarray(probability) * np.asarray(claimed) * policy.recovery_rate - policy.investigation_cost


def rule_points(features: pd.DataFrame) -> pd.Series:
    """Status-quo red-flag scorecard, the baseline every model has to beat."""
    f = features
    return (
        2 * (f["policy_tenure_days"] < 90)
        + 2 * ((f["n_injured"] >= 3) & (f["soft_tissue_only"] == 1))
        + 1 * (f["night_loss"] == 1)
        + 1 * ((f["police_report"] == 0) & f["loss_type"].isin(COLLISIONS))
        + 1 * (f["represented"] == 1)
        + 1 * (f["report_delay_days"] > 14)
        + 1 * (f["prior_claims_3y"] >= 2)
        + 1 * (f["repair_to_value"].fillna(0) > 0.6)
        + 1 * ((f["loss_type"] == "theft") & (f["policy_tenure_days"] < 180))
    ).astype(int)


def rules_priority(features: pd.DataFrame, claimed: pd.Series) -> np.ndarray:
    # Points first; among equal points the larger claim is opened first.
    return rule_points(features).to_numpy() + np.asarray(claimed) / 1e7


def select(priority: np.ndarray, capacity: int, floor: float | None = None) -> np.ndarray:
    """Pick the top `capacity` claims by priority, optionally skipping those at or below `floor`."""
    priority = np.asarray(priority, dtype=float)
    chosen = np.zeros(len(priority), dtype=bool)
    if capacity <= 0 or len(priority) == 0:
        return chosen
    order = np.argsort(-priority, kind="stable")[:capacity]
    chosen[order] = True
    if floor is not None:
        chosen &= priority > floor
    return chosen


def outcome(frame: pd.DataFrame, chosen: np.ndarray, policy: SiuPolicy) -> dict[str, float]:
    """What a set of investigations would have achieved, given the (hindsight) labels."""
    is_fraud = frame["is_fraud"].to_numpy() == 1
    fraud_amount = frame["fraud_amount"].to_numpy()
    caught = chosen & is_fraud
    investigated = int(chosen.sum())
    caught_dollars = float(fraud_amount[caught].sum())
    total_dollars = float(fraud_amount.sum())
    prevented = policy.recovery_rate * caught_dollars
    cost = investigated * policy.investigation_cost
    return {
        "investigated": investigated,
        "frauds_caught": int(caught.sum()),
        "frauds_total": int(is_fraud.sum()),
        "hit_rate": caught.sum() / investigated if investigated else 0.0,
        "fraud_dollars_caught": caught_dollars,
        "fraud_dollars_total": total_dollars,
        "dollar_recall": caught_dollars / total_dollars if total_dollars else 0.0,
        "prevented": prevented,
        "cost": cost,
        "net_savings": prevented - cost,
        "honest_investigated": int((chosen & ~is_fraud).sum()),
    }


def monthly_outcome(frame: pd.DataFrame, priority: np.ndarray, policy: SiuPolicy, capacity: int | None = None,
                    floor: float | None = None) -> dict[str, float]:
    """Apply the capacity month by month (the SIU cannot borrow next month's hours) and average."""
    capacity = policy.capacity if capacity is None else capacity
    priority = np.asarray(priority, dtype=float)
    rows = []
    for _, index in frame.groupby("month").indices.items():
        chosen = select(priority[index], capacity, floor)
        rows.append(outcome(frame.iloc[index], chosen, policy))
    averaged = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    # Rates are recomputed from the averaged totals rather than averaged as rates.
    averaged["hit_rate"] = averaged["frauds_caught"] / averaged["investigated"] if averaged["investigated"] else 0.0
    averaged["dollar_recall"] = averaged["fraud_dollars_caught"] / averaged["fraud_dollars_total"] if averaged["fraud_dollars_total"] else 0.0
    averaged["months"] = len(rows)
    return averaged


def random_outcome(frame: pd.DataFrame, policy: SiuPolicy, capacity: int | None = None) -> dict[str, float]:
    """Expected result of picking files at random, computed analytically per month."""
    capacity = policy.capacity if capacity is None else capacity
    rows = []
    for _, month in frame.groupby("month"):
        n = len(month)
        k = min(capacity, n)
        caught_dollars = k / n * month["fraud_amount"].sum()
        rows.append({"net_savings": policy.recovery_rate * caught_dollars - k * policy.investigation_cost,
                     "fraud_dollars_caught": caught_dollars, "frauds_caught": k / n * month["is_fraud"].sum(),
                     "investigated": k})
    return pd.DataFrame(rows).mean().to_dict()


def capacity_curve(frame: pd.DataFrame, priorities: dict[str, np.ndarray], policy: SiuPolicy,
                   capacities: list[int]) -> dict[str, list[float]]:
    """Average monthly net savings as the SIU team grows or shrinks."""
    curve: dict[str, list[float]] = {"capacity": capacities}
    for name, priority in priorities.items():
        curve[name] = [monthly_outcome(frame, priority, policy, capacity)["net_savings"] for capacity in capacities]
    curve["random"] = [random_outcome(frame, policy, capacity)["net_savings"] for capacity in capacities]
    return curve


def risk_band(probability: np.ndarray, in_queue: np.ndarray, policy: SiuPolicy) -> np.ndarray:
    probability = np.asarray(probability)
    band = np.where(probability < policy.fast_track_below, "Fast-track", "Standard")
    band = np.where(probability >= policy.review_above, "Review", band)
    return np.where(in_queue, "Investigate", band)
