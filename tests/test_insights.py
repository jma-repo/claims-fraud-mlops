import numpy as np
import pandas as pd

from fraudops.insights import categorical_psi, find_rings, overrepresented, psi, tail_shift


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert psi(reference, rng.normal(0, 1, 5000)) < 0.02
    assert psi(reference, rng.normal(1.5, 1, 5000)) > 0.25


def test_categorical_psi_detects_a_new_mix():
    before = pd.Series(["a"] * 80 + ["b"] * 20)
    assert categorical_psi(before, before) < 1e-9
    assert categorical_psi(before, pd.Series(["a"] * 20 + ["b"] * 80)) > 0.25


def test_tail_shift_catches_what_psi_averages_away():
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.exponential(2, 8000))
    current = pd.Series(np.r_[rng.exponential(2, 610), rng.uniform(40, 90, 32)])  # 5% extreme newcomers
    ratio, above = tail_shift(reference, current)
    assert psi(reference, current) < 0.1          # PSI barely notices
    assert ratio > 3 and above >= 30               # the tail check does


def test_density_ratio_flags_a_cluster_that_is_new_this_month():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, (3000, 5))
    current = np.vstack([rng.normal(0, 1, (300, 5)), rng.normal(4, 0.3, (30, 5))])
    ratio = overrepresented(reference, current, seed=0)
    assert (ratio[300:] > 4).mean() > 0.9         # the new cluster is flagged
    assert (ratio[:300] > 4).mean() < 0.05        # ordinary claims are not


def test_rings_link_claims_through_shared_identifiers():
    window = pd.DataFrame({
        "claim_id": [f"C{i}" for i in range(7)],
        "phone_id": ["P1", "P1", "P2", "P2", "P9", "P8", "P7"],
        "address_id": ["A1", "A2", "A2", "A3", "A9", "A8", "A7"],
        "month": ["2026-01"] * 7, "total_claimed": [1000.0] * 7, "city": ["Calgary"] * 7,
        "clinic_id": ["CL1"] * 4 + [None] * 3, "rep_id": [None] * 7, "body_shop_id": [None] * 7, "tow_company_id": [None] * 7,
        "n_injured": [3] * 7, "probability": [0.5] * 7, "is_current": [False] * 7, "is_fraud": [1] * 4 + [0] * 3,
        "ring_id": ["R"] * 4 + [None] * 3,
    })
    rings, networks = find_rings(window, min_claims=4)
    assert len(rings) == 1
    assert rings[0]["claim_ids"] == ["C0", "C1", "C2", "C3"]
    assert rings[0]["shared_phones"] == 2 and rings[0]["shared_addresses"] == 1
    assert networks and networks[0]["ring"] == "R1"
