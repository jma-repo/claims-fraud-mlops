"""Synthetic auto-insurance claims with realistic, overlapping fraud patterns.

Nothing here is real. Providers and people are coded IDs (SHOP-014, PH-4821907) so no real
business is ever tied to fraud, and every distribution is an assumption written down in code.
The fraud mechanisms follow publicly described SIU red flags: staged-collision rings, repair
build-up, injury build-up, owner give-ups, post-loss policy inception, towing/storage schemes
and soft padding of honest losses. The towing scheme deliberately appears only at the end of
the history and grows in the newest month, so the pipeline has a genuine "new pattern the
model has never seen".

Legitimate claims carry many of the same red flags (families with several injured
passengers, new customers, lawyers on real injuries), which keeps the problem honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudops.config import Settings
from fraudops.features import derive_claim_fields

# City -> (province, share of claims, rural). Only provinces where private carriers write auto.
CITIES: dict[str, tuple[str, float, bool]] = {
    "Toronto": ("ON", 0.17, False),
    "Mississauga": ("ON", 0.08, False),
    "Markham": ("ON", 0.05, False),
    "Hamilton": ("ON", 0.06, False),
    "Ottawa": ("ON", 0.08, False),
    "London": ("ON", 0.05, False),
    "Calgary": ("AB", 0.17, False),
    "Edmonton": ("AB", 0.14, False),
    "Red Deer": ("AB", 0.03, True),
    "Lethbridge": ("AB", 0.03, True),
    "Halifax": ("NS", 0.09, False),
    "Moncton": ("NB", 0.05, True),
}

ROADS: dict[str, list[str]] = {
    "Toronto": ["Highway 401", "the Gardiner Expressway", "Yonge Street", "the Don Valley Parkway"],
    "Mississauga": ["Hurontario Street", "Highway 403", "Dundas Street West"],
    "Markham": ["Highway 7", "Highway 404", "Woodbine Avenue"],
    "Hamilton": ["the Lincoln Alexander Parkway", "Main Street East", "the QEW"],
    "Ottawa": ["Highway 417", "Bank Street", "Hunt Club Road"],
    "London": ["Wellington Road", "Highway 401", "Oxford Street"],
    "Calgary": ["Deerfoot Trail", "Macleod Trail", "Stoney Trail", "Crowchild Trail"],
    "Edmonton": ["the Anthony Henday Drive", "Whitemud Drive", "Gateway Boulevard"],
    "Red Deer": ["Highway 2", "Gaetz Avenue", "Highway 11"],
    "Lethbridge": ["Highway 3", "Mayor Magrath Drive", "Highway 4"],
    "Halifax": ["Highway 102", "Barrington Street", "the Bedford Highway"],
    "Moncton": ["Highway 15", "Mountain Road", "Wheeler Boulevard"],
}

PLACES = ["a grocery store lot", "the insured's driveway", "a condo parkade", "a shopping centre",
          "a transit park-and-ride", "a gym parking lot", "the street in front of the residence"]

LOSS_TYPES = ["rear_end", "intersection", "single_vehicle", "sideswipe", "parking_lot",
              "theft", "hail", "glass", "vandalism", "animal"]
COLLISIONS = {"rear_end", "intersection", "single_vehicle", "sideswipe", "animal"}

REPAIR_MEDIAN = {"rear_end": 4200, "intersection": 7500, "single_vehicle": 8500, "sideswipe": 3200,
                 "parking_lot": 1900, "hail": 5200, "glass": 650, "vandalism": 1600, "animal": 5800}
INJURY_RATE = {"rear_end": 0.28, "intersection": 0.32, "single_vehicle": 0.22, "sideswipe": 0.08,
               "parking_lot": 0.02, "animal": 0.07}
TOW_RATE = {"rear_end": 0.30, "intersection": 0.55, "single_vehicle": 0.65, "sideswipe": 0.12, "animal": 0.50}
POLICE_RATE = {"rear_end": 0.55, "intersection": 0.75, "single_vehicle": 0.60, "sideswipe": 0.40,
               "parking_lot": 0.15, "theft": 0.97, "hail": 0.02, "glass": 0.01, "vandalism": 0.55, "animal": 0.35}
LAWYER_RATE = {"ON": 0.35, "AB": 0.18, "NS": 0.12, "NB": 0.12}

SCHEMES = ["staged_ring", "repair_buildup", "injury_buildup", "owner_giveup", "post_loss_inception",
           "towing_storage", "soft_padding"]


@dataclass(frozen=True)
class Ring:
    ring_id: str
    cities: tuple[str, ...]
    start: int          # first active month index (inclusive)
    end: int            # last active month index (exclusive)
    per_month: float
    n_phones: int
    n_addresses: int


RINGS = (
    Ring("RING-A", ("Toronto", "Mississauga"), 0, 25, 3.0, 9, 6),
    Ring("RING-B", ("Calgary",), 6, 25, 2.5, 7, 5),
    Ring("RING-C", ("Edmonton",), 14, 25, 3.0, 6, 4),
)

# Emerging towing/storage scheme: claims per month index. Index 24 is the newest month.
TOWING_RAMP = {22: 2, 23: 5, 24: 32}
TOWING_CITIES = ("Toronto", "Mississauga", "Hamilton", "Markham")
FAR_YARDS = ("Barrie", "Kitchener", "St. Catharines", "Oshawa", "Guelph")


class _Directory:
    """Coded provider IDs per city, plus the specific providers each scheme uses."""

    def __init__(self) -> None:
        self.shops: dict[str, list[str]] = {}
        self.clinics: dict[str, list[str]] = {}
        self.tows: dict[str, list[str]] = {}
        shop = clinic = tow = 1
        for city, (_, share, _) in CITIES.items():
            n_shops = 4 if share >= 0.1 else 3
            self.shops[city] = [f"SHOP-{shop + i:03d}" for i in range(n_shops)]
            self.clinics[city] = [f"CLINIC-{clinic + i:02d}" for i in range(2)]
            self.tows[city] = [f"TOW-{tow + i:02d}" for i in range(2)]
            shop, clinic, tow = shop + n_shops, clinic + 2, tow + 2
        self.reps = [f"REP-{i:02d}" for i in range(1, 11)]
        self.ring_providers = {
            "RING-A": dict(shop=self.shops["Toronto"][2], clinic=self.clinics["Toronto"][1],
                           rep="REP-03", tow=self.tows["Mississauga"][0]),
            "RING-B": dict(shop=self.shops["Calgary"][1], clinic=self.clinics["Calgary"][0],
                           rep="REP-07", tow=self.tows["Calgary"][1]),
            "RING-C": dict(shop=self.shops["Edmonton"][3], clinic=self.clinics["Edmonton"][1],
                           rep="REP-09", tow=self.tows["Edmonton"][0]),
        }
        self.buildup_shops = {self.shops["Hamilton"][0], self.shops["Edmonton"][2]}
        self.emerging_tow = f"TOW-{tow:02d}"


def _month_index_to_period(first_month: str, index: int) -> pd.Period:
    return pd.Period(first_month, freq="M") + index


def _loss_type_weights(month_of_year: int, province: str) -> np.ndarray:
    weights = dict(rear_end=0.22, intersection=0.14, single_vehicle=0.09, sideswipe=0.07, parking_lot=0.12,
                   theft=0.05, hail=0.0, glass=0.12, vandalism=0.05, animal=0.04)
    if month_of_year in (12, 1, 2, 3):
        weights["single_vehicle"] *= 1.8
        weights["glass"] *= 1.3
    if province == "AB":
        weights["animal"] *= 1.6
        weights["hail"] = {6: 0.14, 7: 0.14, 8: 0.14, 5: 0.04, 9: 0.04}.get(month_of_year, 0.0)
    elif month_of_year in (6, 7, 8):
        weights["hail"] = 0.01
    values = np.array([weights[name] for name in LOSS_TYPES])
    return values / values.sum()


class _Generator:
    def __init__(self, settings: Settings, n_months: int | None = None, claims_per_month: int | None = None):
        cfg = settings.pipeline["data"]
        self.rng = np.random.default_rng(settings.seed)
        self.first_month = cfg["first_month"]
        self.n_months = n_months if n_months is not None else int(cfg["history_months"]) + 1
        self.claims_per_month = claims_per_month or int(cfg["claims_per_month"])
        self.scale = self.claims_per_month / int(cfg["claims_per_month"])
        self.dir = _Directory()
        self.city_names = list(CITIES)
        city_shares = np.array([CITIES[c][1] for c in self.city_names])
        self.city_p = city_shares / city_shares.sum()
        self.claimants: dict[str, list[tuple[str, str, str]]] = {city: [] for city in CITIES}
        self.ring_pools = {
            ring.ring_id: ([self._new_phone() for _ in range(ring.n_phones)],
                           [self._new_address() for _ in range(ring.n_addresses)])
            for ring in RINGS
        }
        self.rows: list[dict] = []

    # Identifiers share one format across legitimate and fraudulent claims, so the ID itself
    # can never leak the label.
    def _new_phone(self) -> str:
        return f"PH-{self.rng.integers(1_000_000, 9_999_999)}"

    def _new_address(self) -> str:
        return f"AD-{self.rng.integers(100_000, 999_999)}"

    def _new_policy(self) -> str:
        return f"POL-{self.rng.integers(10_000_000, 99_999_999)}"

    def _pick(self, options: list[str], skew: float = 1.0) -> str:
        weights = np.array([1.0 / (i + 1) ** skew for i in range(len(options))])
        return options[self.rng.choice(len(options), p=weights / weights.sum())]

    def _lognormal(self, median: float, sigma: float) -> float:
        return float(median * np.exp(self.rng.normal(0, sigma)))

    def _tow_fee(self, km: float, storage_days: int) -> float:
        """Hook fee + mileage + daily storage, with the spread real invoices have."""
        return (175 + 4.2 * km + 75 * storage_days) * float(np.exp(self.rng.normal(0, 0.18)))

    def _loss_hour(self) -> int:
        weights = np.array([1, 0.6, 0.5, 0.4, 0.5, 1.2, 2.5, 4, 4.5, 3, 2.5, 2.6,
                            3, 2.8, 3, 3.8, 4.6, 4.8, 3.6, 2.6, 2, 1.7, 1.5, 1.2])
        return int(self.rng.choice(24, p=weights / weights.sum()))

    def _claimant(self, city: str) -> tuple[str, str, str]:
        roll = self.rng.random()
        known = self.claimants[city]
        if known and roll < 0.07:   # repeat claimant: same person, same policy
            return known[self.rng.integers(len(known))]
        if known and roll < 0.11:   # household member: same address, own phone and policy
            _, address, _ = known[self.rng.integers(len(known))]
            person = (self._new_phone(), address, self._new_policy())
        else:
            person = (self._new_phone(), self._new_address(), self._new_policy())
        known.append(person)
        return person

    def _base_claim(self, month: int, city: str | None = None, loss_type: str | None = None) -> dict:
        rng = self.rng
        period = _month_index_to_period(self.first_month, month)
        city = city or self.city_names[rng.choice(len(self.city_names), p=self.city_p)]
        province, _, rural = CITIES[city]
        if loss_type is None:
            loss_type = LOSS_TYPES[rng.choice(len(LOSS_TYPES), p=_loss_type_weights(period.month, province))]

        day = rng.uniform(0, period.days_in_month)
        report_ts = period.start_time + pd.Timedelta(days=day)
        if rng.random() < 0.09:
            report_delay = int(rng.integers(8, 46))
        else:
            report_delay = int(rng.geometric(0.45) - 1)
        tenure = int(rng.integers(1, 121)) if rng.random() < 0.07 else int(rng.integers(121, 4000))
        # Thieves favour newer, pricier vehicles, so legitimate thefts skew expensive too.
        vehicle_value = float(np.clip(self._lognormal(40_000 if loss_type == "theft" else 26_000, 0.5), 3_000, 140_000))
        n_occupants = int(min(1 + rng.poisson(0.45), 5))

        repair = 0.0
        if loss_type in REPAIR_MEDIAN:
            sigma = 0.35 if loss_type == "glass" else 0.6
            repair = min(self._lognormal(REPAIR_MEDIAN[loss_type], sigma), vehicle_value)

        n_injured, soft_tissue, injury = 0, False, 0.0
        if rng.random() < INJURY_RATE.get(loss_type, 0.0):
            n_injured = max(1, int(rng.binomial(n_occupants, 0.6)))
            soft_tissue = bool(rng.random() < 0.65)
            per_person = 8_000 if soft_tissue else 28_000
            injury = sum(self._lognormal(per_person, 0.55) for _ in range(n_injured))

        rep = None
        if n_injured and rng.random() < LAWYER_RATE[province]:
            rep = self._pick(self.dir.reps, 0.6)
        elif rng.random() < 0.01:
            rep = self._pick(self.dir.reps, 0.6)

        tow, tow_km, storage = None, 0.0, 0
        if rng.random() < TOW_RATE.get(loss_type, 0.0):
            tow = self._pick(self.dir.tows[city])
            tow_km = float(np.clip(self._lognormal(28 if rural else 11, 0.5 if rural else 0.55), 1, 110))
            storage = int(rng.integers(8, 26)) if rng.random() < 0.05 else int(rng.geometric(0.35))
        tow_fee = self._tow_fee(tow_km, storage) if tow else 0.0

        phone, address, policy = self._claimant(city)
        return dict(
            month=str(period),
            report_ts=report_ts.floor("s"),
            province=province,
            city=city,
            loss_type=loss_type,
            loss_hour=self._loss_hour(),
            report_delay_days=report_delay,
            policy_tenure_days=tenure,
            policy_id=policy,
            vehicle_value=round(vehicle_value, 2),
            vehicle_age=int(np.clip(rng.normal(7, 4), 0, 25)),
            n_occupants=n_occupants,
            n_injured=n_injured,
            soft_tissue_only=soft_tissue,
            injury_reserve=round(injury, 2),
            police_report=bool(rng.random() < POLICE_RATE[loss_type]),
            witnesses=int(rng.poisson(0.7)) if loss_type in COLLISIONS else 0,
            claimant_age=int(np.clip(rng.normal(46, 15), 18, 88)),
            prior_claims_3y=int(rng.poisson(0.3)),
            body_shop_id=self._pick(self.dir.shops[city]) if loss_type in REPAIR_MEDIAN and loss_type != "glass" else None,
            clinic_id=self._pick(self.dir.clinics[city], 0.3) if n_injured else None,
            rep_id=rep,
            tow_company_id=tow,
            tow_distance_km=round(tow_km, 1),
            storage_days=storage,
            tow_fee=round(max(tow_fee, 0.0), 2),
            repair_estimate=round(repair, 2),
            phone_id=phone,
            address_id=address,
            is_fraud=0,
            scheme="none",
            ring_id=None,
            fraud_amount=0.0,
            cues=[],
        )

    # --- fraud mechanisms -------------------------------------------------------------------

    def _staged(self, month: int, ring: Ring) -> dict:
        rng = self.rng
        city = ring.cities[rng.integers(len(ring.cities))]
        claim = self._base_claim(month, city, "rear_end" if rng.random() < 0.75 else "intersection")
        providers = self.dir.ring_providers[ring.ring_id]
        phones, addresses = self.ring_pools[ring.ring_id]
        n = int(rng.choice([3, 4, 5], p=[0.35, 0.4, 0.25]))
        injured = n if rng.random() < 0.75 else n - 1
        claim.update(
            loss_hour=int(rng.integers(19, 24)) if rng.random() < 0.6 else (int(rng.integers(0, 3)) if rng.random() < 0.25 else claim["loss_hour"]),
            report_delay_days=int(rng.integers(0, 3)),
            policy_tenure_days=int(rng.integers(5, 151)) if rng.random() < 0.5 else int(rng.integers(151, 2500)),
            n_occupants=n,
            n_injured=injured,
            soft_tissue_only=bool(rng.random() < 0.92),
            injury_reserve=round(sum(self._lognormal(16_000, 0.35) for _ in range(injured)), 2),
            repair_estimate=round(min(self._lognormal(4_000, 0.45), claim["vehicle_value"]), 2),
            police_report=bool(rng.random() < 0.4),
            witnesses=0 if rng.random() < 0.7 else int(rng.poisson(0.6)),
            claimant_age=int(np.clip(rng.normal(31, 7), 19, 60)),
            prior_claims_3y=int(rng.poisson(1.1)),
            body_shop_id=providers["shop"] if rng.random() < 0.8 else claim["body_shop_id"],
            clinic_id=providers["clinic"] if rng.random() < 0.85 else self._pick(self.dir.clinics[city]),
            rep_id=providers["rep"] if rng.random() < 0.85 else None,
        )
        if rng.random() < 0.6:
            km = float(np.clip(self._lognormal(9, 0.5), 1, 40))
            days = int(rng.geometric(0.3))
            claim.update(tow_company_id=providers["tow"] if rng.random() < 0.7 else self._pick(self.dir.tows[city]),
                         tow_distance_km=round(km, 1), storage_days=days,
                         tow_fee=round(self._tow_fee(km, days), 2))
        if rng.random() < 0.7:
            claim["phone_id"] = phones[rng.integers(len(phones))]
        if rng.random() < 0.65:
            claim["address_id"] = addresses[rng.integers(len(addresses))]
        claim.update(is_fraud=1, scheme="staged_ring", ring_id=ring.ring_id)
        claim["cues"] = [c for c, p in [("sudden_stop", 0.5), ("all_occupants_pain", 0.45), ("clinic_referral", 0.45),
                                        ("quick_representation", 0.35)] if rng.random() < p]
        return claim

    def _towing(self, month: int) -> dict:
        rng = self.rng
        city = TOWING_CITIES[rng.integers(len(TOWING_CITIES))]
        claim = self._base_claim(month, city, ["rear_end", "intersection", "single_vehicle"][rng.integers(3)])
        km = float(rng.uniform(45, 95))
        days = int(rng.integers(14, 46))
        fee = (400 + 8.5 * km + 145 * days) * float(np.exp(rng.normal(0, 0.12)))  # inflated hook, mileage and daily rates
        claim.update(tow_company_id=self.dir.emerging_tow, tow_distance_km=round(km, 1), storage_days=days,
                     tow_fee=round(fee, 2), is_fraud=1, scheme="towing_storage")
        claim["fraud_amount_override"] = fee - (175 + 4.2 * 11 + 75 * 3)
        claim["cues"] = [c for c, p in [("far_storage_yard", 0.85), ("unsolicited_tow", 0.6), ("storage_fees", 0.6)]
                         if rng.random() < p]
        return claim

    def _maybe_opportunistic(self, claim: dict, month: int) -> dict:
        if not self._apply_scheme(claim, month):
            # Legitimate claims mention the same phrases now and then.
            rng = self.rng
            claim["cues"] = [c for c, p in LEGIT_CUE_RATES.get(claim["loss_type"], []) if rng.random() < p]
            if rng.random() < 0.04:
                claim["cues"].append("items_added")
            if claim["n_injured"]:
                claim["cues"] += [c for c, p in [("clinic_referral", 0.3), ("quick_representation", 0.15),
                                                 ("all_occupants_pain", 0.15 if claim["n_injured"] >= 2 else 0.0),
                                                 ("late_injury", 0.06), ("late_passengers", 0.02)] if rng.random() < p]
        return claim

    def _apply_scheme(self, claim: dict, month: int) -> bool:
        """Turn an ordinary claim into one of the opportunistic schemes, or leave it alone.

        A scheme that would add less than $100 (an estimate already at the vehicle's value, say)
        is not applied: a "fraud" label with nothing fraudulent in it would only be label noise.
        """
        rng = self.rng
        month_of_year = _month_index_to_period(self.first_month, month).month
        if claim["body_shop_id"] in self.dir.buildup_shops and claim["repair_estimate"] > 0 and rng.random() < 0.42:
            original = claim["repair_estimate"]
            inflated = min(original * rng.uniform(1.6, 2.6), claim["vehicle_value"])
            if inflated - original < 100:
                return False
            claim.update(repair_estimate=round(inflated, 2), is_fraud=1, scheme="repair_buildup")
            claim["fraud_amount_override"] = inflated - original
            claim["cues"] = [c for c, p in [("supplementary_estimate", 0.45), ("parts_added", 0.3)] if rng.random() < p]
        elif claim["loss_type"] in COLLISIONS and claim["n_injured"] and rng.random() < 0.05:
            added = int(rng.integers(1, 3))
            extra = sum(self._lognormal(11_000, 0.4) for _ in range(added))
            claim.update(n_occupants=min(claim["n_occupants"] + added, 6), n_injured=claim["n_injured"] + added,
                         injury_reserve=round(claim["injury_reserve"] + extra, 2),
                         report_delay_days=int(rng.integers(10, 46)) if rng.random() < 0.7 else claim["report_delay_days"],
                         soft_tissue_only=bool(rng.random() < 0.8),
                         rep_id=claim["rep_id"] or (self._pick(self.dir.reps) if rng.random() < 0.6 else None),
                         is_fraud=1, scheme="injury_buildup")
            claim["fraud_amount_override"] = extra
            claim["cues"] = [c for c, p in [("late_passengers", 0.4), ("late_injury", 0.5)] if rng.random() < p]
        elif claim["loss_type"] == "theft" and rng.random() < 0.12:
            claim.update(vehicle_value=round(float(np.clip(self._lognormal(52_000, 0.35), 15_000, 140_000)), 2),
                         policy_tenure_days=int(rng.integers(10, 111)) if rng.random() < 0.6 else claim["policy_tenure_days"],
                         report_delay_days=int(rng.integers(1, 6)) if rng.random() < 0.4 else claim["report_delay_days"],
                         prior_claims_3y=int(rng.poisson(0.6)), is_fraud=1, scheme="owner_giveup")
            claim["cues"] = [c for c, p in [("keys_accounted", 0.45), ("recent_purchase", 0.35), ("missing_documents", 0.25)]
                             if rng.random() < p]
        elif claim["loss_type"] == "hail" and claim["province"] == "AB" and 5 <= month_of_year <= 9 and rng.random() < 0.07:
            claim.update(policy_tenure_days=int(rng.integers(2, 21)), report_delay_days=int(rng.integers(15, 61)),
                         is_fraud=1, scheme="post_loss_inception")
            claim["cues"] = [c for c, p in [("weathered_damage", 0.35)] if rng.random() < p]
        elif claim["loss_type"] not in ("glass", "theft") and rng.random() < 0.02:
            # Soft fraud: an honest loss padded by 15-35%. Common, small, and nearly invisible,
            # which is what keeps any real fraud model far from perfect.
            pad = rng.uniform(0.15, 0.35)
            pad_injury = claim["injury_reserve"] > 0 and rng.random() < 0.5
            repair_room = min(claim["repair_estimate"] * pad, claim["vehicle_value"] - claim["repair_estimate"])
            if not pad_injury and repair_room < 100:
                pad_injury = claim["injury_reserve"] > 0
            if pad_injury:
                extra = claim["injury_reserve"] * pad
                claim["injury_reserve"] = round(claim["injury_reserve"] + extra, 2)
            elif repair_room >= 100:
                extra = repair_room
                claim["repair_estimate"] = round(claim["repair_estimate"] + extra, 2)
            else:
                return False
            claim.update(is_fraud=1, scheme="soft_padding")
            claim["fraud_amount_override"] = extra
            claim["cues"] = [c for c, p in [("items_added", 0.2), ("supplementary_estimate", 0.2)] if rng.random() < p]
        else:
            return False
        return True

    # --- assembly -----------------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        rng = self.rng
        for month in range(self.n_months):
            period = _month_index_to_period(self.first_month, month)
            seasonal = 1.1 if period.month in (12, 1, 2) else 1.0
            batch: list[dict] = []
            for ring in RINGS:
                if ring.start <= month < ring.end:
                    batch += [self._staged(month, ring) for _ in range(rng.poisson(ring.per_month * self.scale))]
            batch += [self._towing(month) for _ in range(int(round(TOWING_RAMP.get(month, 0) * self.scale)))]
            n_total = rng.poisson(self.claims_per_month * seasonal)
            batch += [self._maybe_opportunistic(self._base_claim(month), month) for _ in range(max(n_total - len(batch), 0))]
            self.rows += batch
        return self._finalize(pd.DataFrame(self.rows))

    def _finalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("report_ts", kind="stable").reset_index(drop=True)
        frame.insert(0, "claim_id", [f"C-{100001 + i}" for i in range(len(frame))])
        frame = derive_claim_fields(frame)
        override = frame.pop("fraud_amount_override") if "fraud_amount_override" in frame else pd.Series(np.nan, index=frame.index)
        whole = frame["scheme"].isin(["staged_ring", "owner_giveup", "post_loss_inception"])
        frame["fraud_amount"] = np.where(whole, frame["total_claimed"], override.fillna(0.0)).round(2)
        frame["narrative"] = [self._narrative(row) for row in frame.itertuples(index=False)]
        frame = frame.drop(columns=["cues"])
        return frame

    def _narrative(self, row) -> str:
        rng = self.rng
        road = ROADS[row.city][rng.integers(len(ROADS[row.city]))]
        place = PLACES[rng.integers(len(PLACES))]
        cues = set(row.cues)
        # Missing provider IDs come back from the frame as NaN, which is truthy.
        row = row._replace(**{name: None for name in ("body_shop_id", "clinic_id", "rep_id", "tow_company_id")
                              if pd.isna(getattr(row, name))})
        parts: list[str] = []

        opening = {
            "rear_end": ["Insured was stopped in traffic on {road} when a third-party vehicle struck the rear bumper.",
                         "Rear-end collision at low speed on {road}; third party accepted fault at the scene.",
                         "Insured's vehicle was hit from behind while waiting to turn on {road}."],
            "intersection": ["Collision at an intersection on {road}; parties dispute who had the right of way.",
                             "Insured was turning left on {road} when the other vehicle entered the intersection."],
            "single_vehicle": ["Insured lost control on an icy stretch of {road} and struck a guardrail.",
                               "Single-vehicle loss on {road}; vehicle left the roadway and hit a pole."],
            "sideswipe": ["Insured was sideswiped while merging on {road}.",
                          "Vehicle was clipped by a lane-changing truck on {road}."],
            "parking_lot": ["Vehicle was struck while parked at {place}; no note was left.",
                            "Low-speed contact in {place}; other driver left contact details."],
            "theft": ["Insured reports the vehicle stolen from {place} overnight.",
                      "Vehicle was taken from {place}; police report filed the same day."],
            "hail": ["Hail damage to hood, roof and trunk after a storm in {city}.",
                     "Insured reports dents across the roof and hood following a hailstorm."],
            "glass": ["Windshield cracked by a stone on {road}; replacement requested.",
                      "Chip in the windshield spread overnight; glass claim opened."],
            "vandalism": ["Vehicle was keyed and a mirror broken while parked at {place}.",
                          "Rear window smashed and items taken while parked at {place}."],
            "animal": ["Collision with a deer on {road} at dusk; front-end damage.",
                       "Insured struck a moose crossing {road}; vehicle not driveable."],
        }[row.loss_type]
        if "sudden_stop" in cues:
            opening = ["Insured states the vehicle ahead stopped suddenly on {road} and contact could not be avoided.",
                       "Lead vehicle braked abruptly for no apparent reason on {road}; low-speed impact."]
        parts.append(opening[rng.integers(len(opening))].format(road=road, place=place, city=row.city))

        if row.n_injured:
            if "all_occupants_pain" in cues:
                parts.append(f"All {row.n_occupants} occupants reported neck and back pain after the collision.")
            elif row.n_injured == 1:
                parts.append(["Driver reported neck stiffness the next day and saw a family doctor.",
                              "One occupant was assessed at hospital and released the same day.",
                              "Driver reported back pain and attended a walk-in clinic."][rng.integers(3)])
            else:
                parts.append([f"{row.n_injured} occupants reported injuries.",
                              f"Driver and {row.n_injured - 1} passenger(s) were assessed by paramedics.",
                              f"{row.n_injured} people in the vehicle sought treatment in the following days."][rng.integers(3)])
            if "clinic_referral" in cues and row.clinic_id:
                parts.append(f"Occupants were referred to {row.clinic_id} and treatment plans were submitted within days.")
            if "quick_representation" in cues and row.rep_id:
                parts.append(f"Claimants retained {row.rep_id} shortly after the loss.")
            if "late_passengers" in cues:
                parts.append("Additional passengers came forward after the initial report.")
            if "late_injury" in cues:
                parts.append("Injury was first reported several weeks after the loss.")

        if row.loss_type in COLLISIONS:
            parts.append("Police attended and a report number was provided." if row.police_report
                         else "No police attendance; reported at a collision reporting centre.")

        if row.tow_company_id:
            if "far_storage_yard" in cues:
                yard = FAR_YARDS[rng.integers(len(FAR_YARDS))]
                parts.append(f"Vehicle towed by {row.tow_company_id} to a storage yard in {yard}.")
            else:
                destination = row.body_shop_id or "the insured's home"
                parts.append(f"Vehicle towed by {row.tow_company_id} to {destination}.")
            if "unsolicited_tow" in cues:
                parts.append("Tow truck arrived at the scene before police were called.")
            if "storage_fees" in cues:
                parts.append("Storage charges are accruing and a release fee was demanded.")
        elif row.body_shop_id:
            parts.append(f"Estimate received from {row.body_shop_id}.")

        if "supplementary_estimate" in cues:
            parts.append("A supplementary estimate was submitted after teardown.")
        if "parts_added" in cues:
            parts.append("Additional parts were added that were not visible in the first-notice photos.")
        if "items_added" in cues:
            parts.append("Insured added further damaged items after the first report.")
        if "keys_accounted" in cues:
            parts.append("Insured states both keys are in their possession.")
        if "recent_purchase" in cues:
            parts.append("Vehicle was purchased recently.")
        if "missing_documents" in cues:
            parts.append("Insured has not yet provided ownership documents.")
        if "weathered_damage" in cues:
            parts.append("Appraiser notes some dents show weathering inconsistent with a recent storm.")

        closers = ["Photos received.", "Adjuster spoke with the insured by phone.", "File acknowledged.", ""]
        parts.append(closers[rng.integers(len(closers))])
        return " ".join(p for p in parts if p)


LEGIT_CUE_RATES: dict[str, list[tuple[str, float]]] = {
    "rear_end": [("sudden_stop", 0.15), ("supplementary_estimate", 0.12), ("parts_added", 0.04)],
    "intersection": [("supplementary_estimate", 0.15), ("parts_added", 0.04)],
    "single_vehicle": [("supplementary_estimate", 0.15), ("parts_added", 0.04), ("unsolicited_tow", 0.03)],
    "sideswipe": [("supplementary_estimate", 0.12)],
    "animal": [("supplementary_estimate", 0.12)],
    "theft": [("keys_accounted", 0.3), ("recent_purchase", 0.12), ("missing_documents", 0.05)],
    "hail": [("weathered_damage", 0.05), ("supplementary_estimate", 0.1)],
}


def generate_claims(settings: Settings, n_months: int | None = None, claims_per_month: int | None = None) -> pd.DataFrame:
    """Generate the full claim history plus the newest month."""
    return _Generator(settings, n_months, claims_per_month).run()


TRUTH_COLUMNS = ["is_fraud", "scheme", "ring_id", "fraud_amount"]


def run(settings: Settings, n_months: int | None = None, claims_per_month: int | None = None) -> pd.DataFrame:
    frame = generate_claims(settings, n_months, claims_per_month)
    settings.paths.data.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.paths.claims, index=False)
    return frame
