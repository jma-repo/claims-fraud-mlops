"""Insights beyond the score: claim segments, emerging patterns, fraud rings and drift.

The supervised model only knows the fraud it was trained on. These unsupervised views are
how a new scheme gets noticed before anyone has labelled it:
  * segments  - KMeans on the last 12 months; what normal claims look like
  * emerging  - kinds of claims suddenly over-represented this month (density ratio), then grouped
  * rings     - claims linked through shared phone numbers or addresses
  * drift     - population stability index (PSI) per feature and for the score itself
"""

from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fraudops.config import Settings
from fraudops.features import LOSS_TYPES, NUMERIC
from fraudops.model import ModelBundle
from fraudops.text_model import TOKEN_PATTERN, scrub

SEGMENT_FEATURES = [
    "log_total_claimed", "log_injury_reserve", "log_repair_estimate", "n_injured", "injured_share",
    "soft_tissue_only", "represented", "police_report", "total_loss",
    "tow_distance_km", "storage_days", "tow_fee", "policy_tenure_days", "report_delay_days",
]
LOG_FEATURES = {"tow_distance_km", "storage_days", "tow_fee", "policy_tenure_days", "report_delay_days"}
LOSS_WEIGHT = 1.5  # one-hot loss type, weighted so segments stay readable by loss type
SEPARATOR = f" {chr(0xB7)} "  # middle dot, from its code point to keep this file ASCII

TRAIT_LABELS = {  # feature -> (label when high, label when low)
    "n_injured": ("multi-injury", None),
    "injured_share": ("most occupants injured", None),
    "soft_tissue_only": ("soft-tissue", None),
    "represented": ("represented", None),
    "police_report": ("police on scene", "no police report"),
    "night_loss": ("night-time", None),
    "total_loss": ("total losses", None),
    "tow_distance_km": ("long tows", None),
    "storage_days": ("long storage", None),
    "tow_fee": ("towed", None),
    "policy_tenure_days": (None, "new policies"),
    "report_delay_days": ("late-reported", None),
    "log_total_claimed": ("high-value", "low-value"),
    "log_injury_reserve": ("injury-heavy", None),
}
LOSS_LABELS = {"rear_end": "Rear-end", "intersection": "Intersection", "single_vehicle": "Single-vehicle",
               "sideswipe": "Sideswipe", "parking_lot": "Parking-lot", "theft": "Theft", "hail": "Hail",
               "glass": "Glass", "vandalism": "Vandalism", "animal": "Animal strike"}


# --- shared helpers -------------------------------------------------------------------------

def segment_matrix(features: pd.DataFrame, scaler: StandardScaler | None = None,
                   loss_weight: float = LOSS_WEIGHT) -> tuple[np.ndarray, StandardScaler]:
    numeric = features[SEGMENT_FEATURES].astype(float).fillna(0.0).copy()
    for column in LOG_FEATURES:
        numeric[column] = np.log1p(numeric[column].clip(lower=0))
    if scaler is None:
        scaler = StandardScaler().fit(numeric)
    scaled = scaler.transform(numeric)
    loss = np.stack([(features["loss_type"] == name).to_numpy(float) * loss_weight for name in LOSS_TYPES], axis=1)
    return np.hstack([scaled, loss]), scaler


def _traits(z_scores: pd.Series, limit: int = 2) -> list[str]:
    traits = []
    for feature, z in z_scores.reindex(z_scores.abs().sort_values(ascending=False).index).items():
        high, low = TRAIT_LABELS.get(feature, (None, None))
        label = high if z >= 0.8 else low if z <= -0.8 else None
        if label and label not in traits:
            traits.append(label)
        if len(traits) == limit:
            break
    return traits


def _top_terms(texts: pd.Series, background: pd.Series, k: int = 6) -> list[str]:
    """Words over-represented in a group's notes compared with all notes."""
    vectorizer = TfidfVectorizer(preprocessor=scrub, token_pattern=TOKEN_PATTERN, stop_words="english",
                                 ngram_range=(1, 2), min_df=2, max_features=4000)
    vectorizer.fit(background)
    group = np.asarray(vectorizer.transform(texts).mean(axis=0)).ravel()
    overall = np.asarray(vectorizer.transform(background).mean(axis=0)).ravel()
    lift = group - overall
    names = vectorizer.get_feature_names_out()
    terms: list[str] = []
    for index in np.argsort(-lift):
        if lift[index] <= 0 or len(terms) == k:
            break
        if not any(names[index] in kept or kept in names[index] for kept in terms):
            terms.append(names[index])
    return terms


def _dominant(values: pd.Series, min_share: float = 0.5) -> dict[str, Any] | None:
    # Share of all members, not of those with a value: one represented claim is not "100%".
    counts = values.dropna().value_counts() / max(len(values), 1)
    if counts.empty or counts.iloc[0] < min_share:
        return None
    return {"value": str(counts.index[0]), "share": round(float(counts.iloc[0]), 3)}


# --- segments and emerging patterns --------------------------------------------------------------

def build_segments(reference: pd.DataFrame, current: pd.DataFrame, k: int, seed: int, novelty_ratio: float):
    ref_x, scaler = segment_matrix(reference)
    cur_x, _ = segment_matrix(current, scaler)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(ref_x)
    ref_label = kmeans.labels_
    cur_label = kmeans.predict(cur_x)
    novel = overrepresented(ref_x, cur_x, seed) >= novelty_ratio
    return ref_x, cur_x, ref_label, cur_label, novel, scaler


def overrepresented(ref_x: np.ndarray, cur_x: np.ndarray, seed: int) -> np.ndarray:
    """Density ratio p(claim | this month) / p(claim | last 12 months), via a domain classifier.

    A classifier learns to tell this month's claims from last year's; its out-of-fold odds,
    rescaled by the class sizes, say how much more common a kind of claim has become. Distance
    or outlier scores miss a new scheme once it has a few early cases in the reference window
    and dozens of members this month; a density ratio does not.
    """
    x = np.vstack([ref_x, cur_x])
    y = np.r_[np.zeros(len(ref_x)), np.ones(len(cur_x))]
    oof = np.zeros(len(x))
    for train_idx, hold_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y):
        classifier = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                        random_state=seed, deterministic=True, force_col_wise=True, verbose=-1)
        classifier.fit(x[train_idx], y[train_idx])
        oof[hold_idx] = classifier.predict_proba(x[hold_idx])[:, 1]
    p = np.clip(oof[len(ref_x):], 1e-6, 1 - 1e-6)
    return (p / (1 - p)) / (len(cur_x) / len(ref_x))


def profile_segments(reference: pd.DataFrame, current: pd.DataFrame, ref_label: np.ndarray,
                     cur_label: np.ndarray, k: int) -> list[dict[str, Any]]:
    numeric = reference[SEGMENT_FEATURES].astype(float)
    mean, std = numeric.mean(), numeric.std().replace(0, 1)
    segments = []
    for s in range(k):
        ref, cur = reference[ref_label == s], current[cur_label == s]
        z = (ref[SEGMENT_FEATURES].astype(float).mean() - mean) / std
        loss = ref["loss_type"].value_counts(normalize=True)
        top_loss = [LOSS_LABELS[name] for name, share in loss.items() if share >= 0.25][:2] or [LOSS_LABELS[loss.index[0]]]
        traits = _traits(z)
        segments.append({
            "id": s,
            "name": " / ".join(top_loss) + (SEPARATOR + SEPARATOR.join(traits) if traits else ""),
            "loss_types": {LOSS_LABELS[name]: round(float(share), 3) for name, share in loss.head(3).items()},
            "traits": traits,
            "claims_reference": int(len(ref)),
            "share_reference": round(len(ref) / len(reference), 4),
            "claims_current": int(len(cur)),
            "share_current": round(len(cur) / max(len(current), 1), 4),
            "avg_claimed": round(float(ref["total_claimed"].mean()), 2),
            "confirmed_fraud_rate": round(float(ref["is_fraud"].mean()), 4),
            "fraud_dollar_share": round(float(ref["fraud_amount"].sum() / max(reference["fraud_amount"].sum(), 1)), 4),
            "avg_probability_current": round(float(cur["probability"].mean()), 4) if len(cur) else None,
            "top_terms": _top_terms(ref["narrative"], reference["narrative"], 5),
            "z_scores": {feature: round(float(value), 2) for feature, value in z.items()},
        })
    return segments


def group_flagged(x: np.ndarray, distance: float, min_size: int = 5) -> np.ndarray:
    """Average-linkage clusters cut at a fixed distance in the scaled feature space; groups smaller
    than `min_size` are left as noise (-1). On a few dozen flagged claims this is steadier than
    density-based clustering, which tends to split one scheme by loss type."""
    groups = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, linkage="average").fit_predict(x)
    sizes = pd.Series(groups).value_counts()
    keep = {g for g, size in sizes.items() if size >= min_size}
    renumber = {g: i for i, g in enumerate(sorted(keep, key=lambda g: -sizes[g]))}
    return np.array([renumber.get(g, -1) for g in groups])


def emerging_patterns(reference: pd.DataFrame, current: pd.DataFrame, cur_x: np.ndarray, novel: np.ndarray,
                      history: pd.DataFrame, months: list[str], distance: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Group out-of-pattern claims; a group that is dense and shares providers is a candidate scheme."""
    labels = np.full(len(current), -1)
    if novel.sum() < 5:
        return [], labels
    groups = group_flagged(cur_x[novel], distance)
    labels[np.flatnonzero(novel)] = groups
    numeric = reference[SEGMENT_FEATURES].astype(float)
    mean, std = numeric.mean(), numeric.std().replace(0, 1)
    patterns = []
    for g in sorted(set(groups) - {-1}):
        members = current[labels == g]
        z = (members[SEGMENT_FEATURES].astype(float).mean() - mean) / std
        providers = {kind: _dominant(members[column]) for kind, column in
                     [("tow_company", "tow_company_id"), ("body_shop", "body_shop_id"), ("clinic", "clinic_id"),
                      ("representative", "rep_id")]}
        providers = {kind: value for kind, value in providers.items() if value}
        trend = None
        if providers:
            kind, info = next(iter(providers.items()))
            column = {"tow_company": "tow_company_id", "body_shop": "body_shop_id", "clinic": "clinic_id",
                      "representative": "rep_id"}[kind]
            monthly = history[history[column] == info["value"]].groupby("month").size()
            trend = {"provider": info["value"], "months": months,
                     "claims": [int(monthly.get(month, 0)) for month in months]}
        top_z = z.reindex(z.abs().sort_values(ascending=False).index).head(4)
        patterns.append({
            "id": f"E{g + 1}",
            "claims": int(len(members)),
            "total_claimed": round(float(members["total_claimed"].sum()), 2),
            "avg_probability": round(float(members["probability"].mean()), 4),
            "in_queue": int(members["in_queue"].sum()),
            "cities": members["city"].value_counts().head(4).to_dict(),
            "loss_types": {LOSS_LABELS[k]: int(v) for k, v in members["loss_type"].value_counts().items()},
            "shared_providers": providers,
            "distinctive": [{"feature": feature, "z": round(float(value), 1),
                             "current_mean": round(float(members[feature].astype(float).mean()), 1),
                             "reference_mean": round(float(mean[feature]), 1)} for feature, value in top_z.items()],
            "top_terms": _top_terms(members["narrative"], current["narrative"], 6),
            "trend": trend,
            "claim_ids": members["claim_id"].tolist(),
            # Synthetic-data only: lets the README show whether the alert was real.
            "hidden_truth_fraud_rate": round(float(members["is_fraud"].mean()), 3),
            "hidden_truth_schemes": members["scheme"].value_counts().to_dict(),
        })
    patterns.sort(key=lambda item: -item["total_claimed"])
    return patterns, labels


# --- rings -----------------------------------------------------------------------------------

def find_rings(window: pd.DataFrame, min_claims: int = 4, top_networks: int = 3, seed: int = 42):
    """Connected components of claims that share a phone number or an address."""
    graph = nx.Graph()
    for row in window.itertuples(index=False):
        graph.add_node(row.claim_id, kind="claim")
        graph.add_edge(row.claim_id, f"PH:{row.phone_id}")
        graph.add_edge(row.claim_id, f"AD:{row.address_id}")
    by_id = window.set_index("claim_id")
    rings = []
    for component in nx.connected_components(graph):
        claim_ids = [node for node in component if not node.startswith(("PH:", "AD:"))]
        if len(claim_ids) < min_claims:
            continue
        members = by_id.loc[claim_ids]
        shared_entities = [node for node in component if node.startswith(("PH:", "AD:")) and graph.degree(node) > 1]
        rings.append({
            "claims": len(claim_ids),
            "claim_ids": sorted(claim_ids),
            "months_active": int(members["month"].nunique()),
            "first_month": str(members["month"].min()),
            "last_month": str(members["month"].max()),
            "total_claimed": round(float(members["total_claimed"].sum()), 2),
            "shared_phones": sum(1 for node in shared_entities if node.startswith("PH:")),
            "shared_addresses": sum(1 for node in shared_entities if node.startswith("AD:")),
            "cities": members["city"].value_counts().head(3).to_dict(),
            "providers": {kind: value for kind, value in
                          [("clinic", _dominant(members["clinic_id"], 0.4)), ("representative", _dominant(members["rep_id"], 0.4)),
                           ("body_shop", _dominant(members["body_shop_id"], 0.4)), ("tow_company", _dominant(members["tow_company_id"], 0.4))]
                          if value},
            "avg_injured": round(float(members["n_injured"].mean()), 2),
            "avg_probability": round(float(members["probability"].mean()), 4),
            "claims_current_month": int(members["is_current"].sum()),
            "hidden_truth_fraud_rate": round(float(members["is_fraud"].mean()), 3),
            "hidden_truth_ring": _dominant(members["ring_id"], 0.5),
        })
    rings.sort(key=lambda ring: -ring["total_claimed"])
    for index, ring in enumerate(rings, start=1):
        ring["id"] = f"R{index}"

    networks = []
    for ring in rings[:top_networks]:
        nodes = set(ring["claim_ids"])
        sub = graph.subgraph(nodes | {n for c in nodes for n in graph.neighbors(c)}).copy()
        # Entities used by a single claim add clutter without showing any link.
        sub.remove_nodes_from([n for n in list(sub) if n.startswith(("PH:", "AD:")) and sub.degree(n) < 2])
        layout = nx.spring_layout(sub, seed=seed, k=0.6 / np.sqrt(max(len(sub), 1)), iterations=200)
        networks.append({
            "ring": ring["id"],
            "nodes": [{"id": n, "kind": "phone" if n.startswith("PH:") else "address" if n.startswith("AD:") else "claim",
                       "x": round(float(layout[n][0]), 4), "y": round(float(layout[n][1]), 4),
                       **({"total_claimed": float(by_id.at[n, "total_claimed"]), "probability": float(by_id.at[n, "probability"]),
                           "month": str(by_id.at[n, "month"]), "current": bool(by_id.at[n, "is_current"])}
                          if not n.startswith(("PH:", "AD:")) else {})}
                      for n in sub.nodes],
            "edges": [[a, b] for a, b in sub.edges],
        })
    return rings, networks


# --- drift -----------------------------------------------------------------------------------

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index with reference-quantile bins (ties collapse to fewer bins)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-4, None), np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(categories, fill_value=0).clip(lower=1e-4)
    cur = current.value_counts(normalize=True).reindex(categories, fill_value=0).clip(lower=1e-4)
    return float(((cur - ref) * np.log(cur / ref)).sum())


SCORE_BINS = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])


def tail_shift(reference: pd.Series, current: pd.Series) -> tuple[float, int]:
    """How much more often the newest month lands above the reference 99th percentile.

    PSI averages over bins, so 30 extreme claims in a month of 650 barely move it; a new scheme
    usually starts exactly like that. 1.0 means "as often as before"."""
    reference, current = reference.astype(float).dropna(), current.astype(float).dropna()
    q99 = reference.quantile(0.99)
    if len(current) == 0 or q99 <= reference.median():
        return 1.0, 0
    above = int((current > q99).sum())
    expected = max(float((reference > q99).mean()), 1e-3)
    return above / len(current) / expected, above


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, ref_scores: np.ndarray, cur_scores: np.ndarray,
                 cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for feature in NUMERIC + ["narrative_risk"]:
        ref, cur = reference[feature].astype(float), current[feature].astype(float)
        ratio, above = tail_shift(ref, cur)
        rows.append({
            "feature": feature,
            "psi": round(psi(ref, cur, cfg["bins"]), 4),
            "mean_shift_sd": round(float((cur.mean() - ref.mean()) / (ref.std() or 1.0)), 3),
            "tail_ratio": round(ratio, 2),
            "tail_claims": above,
            "reference_mean": round(float(ref.mean()), 3),
            "current_mean": round(float(cur.mean()), 3),
        })
    rows.append({"feature": "loss_type", "psi": round(categorical_psi(reference["loss_type"], current["loss_type"]), 4),
                 "mean_shift_sd": None, "tail_ratio": None, "tail_claims": None, "reference_mean": None, "current_mean": None})
    for row in rows:
        shifted = row["mean_shift_sd"] is None or abs(row["mean_shift_sd"]) >= 0.2
        tail = (row["tail_ratio"] or 0) >= 3 and (row["tail_claims"] or 0) >= 10
        if (row["psi"] >= cfg["alert"] and shifted) or tail:
            row["status"] = "alert"
        elif row["psi"] >= cfg["warn"] or (row["tail_ratio"] or 0) >= 2:
            row["status"] = "warn"
        else:
            row["status"] = "stable"
    order = {"alert": 0, "warn": 1, "stable": 2}
    rows.sort(key=lambda row: (order[row["status"]], -row["psi"]))

    def shares(scores: np.ndarray) -> list[float]:
        return (np.histogram(np.clip(scores, 0, 1), SCORE_BINS)[0] / len(scores)).round(4).tolist()

    ref_share, cur_share = np.clip(shares(ref_scores), 1e-4, None), np.clip(shares(cur_scores), 1e-4, None)
    return {
        "thresholds": {"warn": cfg["warn"], "alert": cfg["alert"], "tail_ratio": 3},
        "features": rows,
        "score": {
            "psi": round(float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share))), 4),
            "bins": SCORE_BINS.tolist(),
            "reference": shares(ref_scores),
            "current": shares(cur_scores),
        },
        "alerts": [row["feature"] for row in rows if row["status"] == "alert"],
    }


# --- views for the dashboard ------------------------------------------------------------------

MAP_LOSS_WEIGHT = 0.8  # lighter than for clustering, so the map shows profiles, not one island per loss type


def map_coordinates(reference: pd.DataFrame, current: pd.DataFrame, seed: int) -> np.ndarray:
    """2-D t-SNE layout. Binary features create exact duplicates that t-SNE shatters into dozens
    of islands; a little jitter and a high perplexity give a map with readable regions."""
    ref_x, scaler = segment_matrix(reference, loss_weight=MAP_LOSS_WEIGHT)
    cur_x, _ = segment_matrix(current, scaler, loss_weight=MAP_LOSS_WEIGHT)
    points = np.vstack([ref_x, cur_x])
    points = points + np.random.default_rng(seed).normal(0, 0.1, points.shape)
    return TSNE(n_components=2, perplexity=100, init="pca", random_state=seed, learning_rate="auto").fit_transform(points)


def pattern_view(pattern: dict[str, Any], reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any] | None:
    """The emerging pattern on its own two most distinctive features, in real units."""
    features = [item["feature"] for item in pattern["distinctive"] if not item["feature"].startswith("log_")][:2]
    if len(features) < 2:
        return None
    members = set(pattern["claim_ids"])
    rows = []
    for frame, is_current in ((reference, False), (current, True)):
        for row in frame[[*features, "claim_id"]].itertuples(index=False):
            x, y = float(getattr(row, features[0])), float(getattr(row, features[1]))
            if x > 0 and y > 0:  # log axes; claims with no towing at all are not part of this picture
                group = "pattern" if row.claim_id in members else "current" if is_current else "reference"
                rows.append((round(x, 2), round(y, 2), group, row.claim_id))
    return {"pattern": pattern["id"], "x_feature": features[0], "y_feature": features[1],
            "x": [r[0] for r in rows], "y": [r[1] for r in rows], "group": [r[2] for r in rows],
            "claim_id": [r[3] for r in rows]}


# --- pipeline step ---------------------------------------------------------------------------

def run(settings: Settings) -> dict[str, Any]:
    paths, cfg, seed = settings.paths, settings.pipeline, settings.seed
    new_month = cfg["data"]["new_month"]
    bundle = ModelBundle.load(paths.model_dir)
    claims = pd.read_parquet(paths.claims)
    features = pd.read_parquet(paths.features)
    scored = pd.read_parquet(paths.scored)

    frame = features.merge(claims[["claim_id", "city", "narrative", "total_claimed", "body_shop_id", "clinic_id",
                                   "rep_id", "tow_company_id", "phone_id", "address_id", "is_fraud", "fraud_amount",
                                   "scheme", "ring_id"]], on="claim_id")
    months = sorted(frame["month"].unique())
    reference_months = [m for m in months if m < new_month][-12:]
    reference = frame[frame["month"].isin(reference_months)].reset_index(drop=True)
    reference["probability"] = bundle.predict(reference, reference["narrative"])
    reference["in_queue"] = False
    current = frame[frame["month"] == new_month].merge(scored[["claim_id", "probability", "in_queue"]], on="claim_id")
    current = current.reset_index(drop=True)

    seg_cfg = cfg["segments"]
    ref_x, cur_x, ref_label, cur_label, novel, _ = build_segments(reference, current, seg_cfg["k"], seed,
                                                                 seg_cfg["novelty_ratio"])
    segments = profile_segments(reference, current, ref_label, cur_label, seg_cfg["k"])
    trend_months = reference_months + [new_month]
    patterns, emerging_label = emerging_patterns(reference, current, cur_x, novel, frame, trend_months,
                                                 float(seg_cfg["group_distance"]))

    # Pattern map: a sample of the reference year plus every claim of the newest month.
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(reference), size=min(seg_cfg["map_sample"], len(reference)), replace=False)
    coords = map_coordinates(reference.iloc[sample], current, seed)
    emerging_id = {index: f"E{label + 1}" for index, label in enumerate(emerging_label) if label >= 0}
    ref_points = reference.iloc[sample]
    points = []
    for position, row in enumerate(ref_points.itertuples(index=False)):
        points.append({"x": round(float(coords[position, 0]), 2), "y": round(float(coords[position, 1]), 2),
                       "claim_id": row.claim_id, "segment": int(ref_label[sample[position]]), "current": False,
                       "loss_type": row.loss_type, "total_claimed": round(float(row.total_claimed), 0),
                       "probability": round(float(row.probability), 3), "confirmed_fraud": bool(row.is_fraud)})
    offset = len(sample)
    for index, row in enumerate(current.itertuples(index=False)):
        points.append({"x": round(float(coords[offset + index, 0]), 2), "y": round(float(coords[offset + index, 1]), 2),
                       "claim_id": row.claim_id, "segment": int(cur_label[index]), "current": True,
                       "novel": bool(novel[index]), "emerging": emerging_id.get(index),
                       "loss_type": row.loss_type, "total_claimed": round(float(row.total_claimed), 0),
                       "probability": round(float(row.probability), 3), "in_queue": bool(row.in_queue)})

    # Rings over the last year plus the newest month.
    window = pd.concat([reference.assign(is_current=False), current.assign(is_current=True)], ignore_index=True)
    rings, networks = find_rings(window, seed=seed)

    # Drift: reference year vs newest month; score drift against the out-of-sample test months.
    test_months = [m for m in months if cfg["split"]["valid_until"] < m < new_month]
    test_frame = frame[frame["month"].isin(test_months)]
    reference_scores = bundle.predict(test_frame, test_frame["narrative"])
    reference = reference.assign(narrative_risk=bundle.narrative_risk(reference["narrative"]))
    current = current.assign(narrative_risk=bundle.narrative_risk(current["narrative"]))
    drift = drift_report(reference, current, reference_scores, current["probability"].to_numpy(), cfg["drift"])

    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "segments.json").write_text(json.dumps({
        "reference_months": [reference_months[0], reference_months[-1]], "current_month": new_month,
        "novel_claims": int(novel.sum()), "segments": segments, "emerging": patterns, "map": points,
        "pattern_view": pattern_view(patterns[0], reference.iloc[sample], current) if patterns else None,
    }, indent=1), encoding="utf-8")
    (paths.reports / "rings.json").write_text(json.dumps({"rings": rings, "networks": networks}, indent=1), encoding="utf-8")
    (paths.reports / "drift.json").write_text(json.dumps(drift, indent=1), encoding="utf-8")
    return {"segments": len(segments), "emerging": len(patterns), "rings": len(rings),
            "novel": int(novel.sum()), "drift_alerts": drift["alerts"]}
