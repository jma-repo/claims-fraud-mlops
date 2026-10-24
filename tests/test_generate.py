import pandas as pd

from fraudops.config import REPO_ROOT, Paths, Settings
from fraudops.features import validate_claims
from fraudops.generate import SCHEMES, generate_claims


def test_generation_is_deterministic(tmp_path):
    settings = Settings(Paths(repo=REPO_ROOT, work=tmp_path))
    first = generate_claims(settings, n_months=4, claims_per_month=60)
    second = generate_claims(settings, n_months=4, claims_per_month=60)
    pd.testing.assert_frame_equal(first, second)


def test_generated_claims_honour_the_contract(small_claims):
    validate_claims(small_claims)
    assert small_claims["claim_id"].is_unique
    assert 0.03 < small_claims["is_fraud"].mean() < 0.12


def test_every_scheme_appears_and_amounts_are_consistent(small_claims):
    fraud = small_claims[small_claims["is_fraud"] == 1]
    assert set(fraud["scheme"]) == set(SCHEMES)
    assert (fraud["fraud_amount"] > 0).all()
    assert (fraud["fraud_amount"] <= fraud["total_claimed"] + 0.01).all()
    assert (small_claims.loc[small_claims["is_fraud"] == 0, "fraud_amount"] == 0).all()


def test_the_towing_scheme_is_new_in_the_last_month(small_claims):
    towing = small_claims[small_claims["scheme"] == "towing_storage"]
    last_month = small_claims["month"].max()
    assert (towing["month"] == last_month).mean() > 0.7


def test_identifiers_do_not_leak_the_label(small_claims):
    # Same ID formats for fraudulent and legitimate claims: the model cannot learn "ring phones look different".
    for column, pattern in [("phone_id", r"^PH-\d{7}$"), ("address_id", r"^AD-\d{6}$")]:
        assert small_claims[column].str.fullmatch(pattern.strip("^$")).all()


def test_providers_are_coded_ids_not_names(small_claims):
    for column in ["body_shop_id", "clinic_id", "rep_id", "tow_company_id"]:
        values = small_claims[column].dropna()
        assert values.str.match(r"^(SHOP|CLINIC|REP|TOW)-\d+$").all()
