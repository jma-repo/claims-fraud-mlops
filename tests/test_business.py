import numpy as np
import pandas as pd
import pytest

from fraudops import business

POLICY = business.SiuPolicy(capacity=2, investigation_cost=1000.0, recovery_rate=0.5, fast_track_below=0.03, review_above=0.25)


def test_expected_value_is_probability_times_recoverable_amount_minus_cost():
    value = business.expected_value(np.array([0.5, 0.01]), np.array([10_000, 10_000]), POLICY)
    np.testing.assert_allclose(value, [0.5 * 10_000 * 0.5 - 1000, 0.01 * 10_000 * 0.5 - 1000])


def test_select_respects_capacity_and_skips_negative_value():
    chosen = business.select(np.array([5.0, -1.0, 9.0, 3.0]), capacity=3, floor=0.0)
    assert chosen.tolist() == [True, False, True, True]
    assert business.select(np.array([-5.0, -1.0]), capacity=2, floor=0.0).sum() == 0
    assert business.select(np.array([]), capacity=3).sum() == 0


def test_outcome_counts_dollars_cost_and_honest_customers():
    frame = pd.DataFrame({"is_fraud": [1, 0, 1, 0], "fraud_amount": [8000.0, 0.0, 2000.0, 0.0]})
    result = business.outcome(frame, np.array([True, True, False, False]), POLICY)
    assert result["frauds_caught"] == 1
    assert result["honest_investigated"] == 1
    assert result["prevented"] == pytest.approx(4000.0)
    assert result["net_savings"] == pytest.approx(4000.0 - 2000.0)
    assert result["dollar_recall"] == pytest.approx(0.8)


def test_capacity_applies_per_month():
    frame = pd.DataFrame({
        "month": ["2026-01"] * 3 + ["2026-02"] * 3,
        "is_fraud": [1, 1, 1, 1, 0, 0],
        "fraud_amount": [1000.0] * 4 + [0.0, 0.0],
    })
    priority = np.array([3, 2, 1, 3, 2, 1], dtype=float)
    result = business.monthly_outcome(frame, priority, POLICY)
    assert result["investigated"] == 2      # two files each month, averaged
    assert result["frauds_caught"] == 1.5   # 2 in January, 1 in February


def test_random_baseline_is_capacity_share_of_fraud_dollars():
    frame = pd.DataFrame({"month": ["m"] * 4, "is_fraud": [1, 0, 0, 0], "fraud_amount": [4000.0, 0, 0, 0]})
    result = business.random_outcome(frame, POLICY)
    assert result["fraud_dollars_caught"] == pytest.approx(2000.0)
    assert result["net_savings"] == pytest.approx(0.5 * 2000.0 - 2 * 1000.0)


def test_risk_bands():
    bands = business.risk_band(np.array([0.01, 0.1, 0.3, 0.3]), np.array([False, False, False, True]), POLICY)
    assert bands.tolist() == ["Fast-track", "Standard", "Review", "Investigate"]


def test_rule_points_reward_classic_red_flags():
    features = pd.DataFrame({
        "policy_tenure_days": [30, 3000], "n_injured": [4, 0], "soft_tissue_only": [1, 0], "night_loss": [1, 0],
        "police_report": [0, 1], "loss_type": ["rear_end", "glass"], "represented": [1, 0], "report_delay_days": [1, 1],
        "prior_claims_3y": [2, 0], "repair_to_value": [0.1, 0.05],
    })
    points = business.rule_points(features)
    assert points.iloc[0] > points.iloc[1] == 0
