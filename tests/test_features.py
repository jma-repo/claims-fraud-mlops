import numpy as np
import pandas as pd
import pytest

from fraudops.config import REPO_ROOT, Paths, Settings
from fraudops.features import (
    NUMERIC,
    DataContractError,
    OnlineFeatureStore,
    compute_features,
    derive_claim_fields,
    feature_sql,
    validate_claims,
)

SQL = feature_sql(Settings(Paths(repo=REPO_ROOT, work=REPO_ROOT)))


def _claims(small_claims: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Tiny frames built on a real row, so every contract column is present."""
    base = small_claims.iloc[0].to_dict()
    frame = pd.DataFrame([{**base, **row} for row in rows])
    return derive_claim_fields(frame)


def test_lookback_counts_never_see_the_claim_itself_or_the_future(small_claims):
    claims = _claims(small_claims, [
        {"claim_id": "A", "report_ts": "2025-01-01 10:00", "phone_id": "PH-1", "address_id": "AD-1"},
        {"claim_id": "B", "report_ts": "2025-03-01 10:00", "phone_id": "PH-1", "address_id": "AD-2"},
        {"claim_id": "C", "report_ts": "2025-06-01 10:00", "phone_id": "PH-1", "address_id": "AD-3"},
        {"claim_id": "D", "report_ts": "2026-08-01 10:00", "phone_id": "PH-1", "address_id": "AD-4"},
    ])
    features = compute_features(claims, SQL).set_index("claim_id")
    # A sees nothing; B sees A; C sees A and B; D is more than 365 days after all of them.
    assert features.loc[["A", "B", "C", "D"], "phone_prior_claims_365d"].tolist() == [0, 1, 2, 0]


def test_claims_reported_at_the_same_second_do_not_see_each_other(small_claims):
    claims = _claims(small_claims, [
        {"claim_id": "A", "report_ts": "2025-01-01 10:00", "phone_id": "PH-9", "address_id": "AD-1"},
        {"claim_id": "B", "report_ts": "2025-01-01 10:00", "phone_id": "PH-9", "address_id": "AD-2"},
    ])
    features = compute_features(claims, SQL)
    assert features["phone_prior_claims_365d"].tolist() == [0, 0]


def test_online_features_match_batch_features(small_claims):
    """Training/serving skew check: scoring claims one at a time reproduces the batch features."""
    last = small_claims["month"].max()
    history = small_claims[small_claims["month"] < last]
    incoming = small_claims[small_claims["month"] == last].sort_values("report_ts").head(25)
    batch = compute_features(small_claims, SQL).set_index("claim_id").loc[incoming["claim_id"]]
    store = OnlineFeatureStore(history, SQL)
    online = pd.concat([store.features_for(incoming.iloc[[i]]) for i in range(len(incoming))]).set_index("claim_id")
    np.testing.assert_allclose(online[NUMERIC].to_numpy(float), batch[NUMERIC].to_numpy(float), rtol=1e-9, atol=1e-9)
    assert store.size() == len(history) + len(incoming)


def test_rescoring_a_claim_replaces_it_instead_of_counting_it_twice(small_claims):
    last = small_claims["month"].max()
    store = OnlineFeatureStore(small_claims[small_claims["month"] < last], SQL)
    claim = small_claims[small_claims["month"] == last].head(1)
    first = store.features_for(claim)
    second = store.features_for(claim)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("change, message", [
    ({"n_injured": 9, "n_occupants": 2}, "n_injured > n_occupants"),
    ({"loss_type": "meteor"}, "unknown loss_type"),
    ({"tow_fee": -5.0}, "negative tow_fee"),
])
def test_data_contract_rejects_bad_claims(small_claims, change, message):
    bad = small_claims.head(5).copy()
    for column, value in change.items():
        bad.loc[bad.index[0], column] = value
    with pytest.raises(DataContractError, match=message):
        validate_claims(bad)


def test_data_contract_rejects_duplicates(small_claims):
    with pytest.raises(DataContractError, match="duplicate claim_id"):
        validate_claims(pd.concat([small_claims.head(3), small_claims.head(1)]))
