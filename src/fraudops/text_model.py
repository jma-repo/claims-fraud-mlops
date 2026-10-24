"""Narrative risk: TF-IDF + logistic regression over adjuster notes.

Its probability becomes one feature of the main model, and its strongest terms become a
human-readable reason ("narrative mentions: stopped suddenly, neck back pain").
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from fraudops.generate import CITIES, FAR_YARDS, ROADS

# Words only: provider codes (SHOP-014) and numbers are left to the network features, so the
# text model learns language rather than memorising which shop a ring used.
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"

# Locations are scrubbed before the text model sees a note. Otherwise road names become a
# proxy for neighbourhoods, which is both a shortcut and a fairness risk. A production system
# would use an NER scrubber; this PoC uses the generator's gazetteer.
_PLACE_NAMES = sorted({road.removeprefix("the ") for roads in ROADS.values() for road in roads}
                      | set(CITIES) | set(FAR_YARDS), key=len, reverse=True)
_PLACE_PATTERN = re.compile("|".join(re.escape(name) for name in _PLACE_NAMES), re.IGNORECASE)


def scrub(text: str) -> str:
    return _PLACE_PATTERN.sub(" ", text).lower()


class NarrativeModel:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.pipeline = self._new_pipeline()

    def _new_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000, sublinear_tf=True,
                                      preprocessor=scrub, token_pattern=TOKEN_PATTERN, stop_words="english")),
            ("clf", LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")),
        ])

    def fit(self, texts: pd.Series, labels: pd.Series) -> NarrativeModel:
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: pd.Series | list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(pd.Series(texts))[:, 1]

    def out_of_fold(self, texts: pd.Series, labels: pd.Series, folds: int = 5) -> np.ndarray:
        """Predictions for the training rows from models that never saw them (no stacking leakage)."""
        result = np.zeros(len(texts))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for train_idx, hold_idx in splitter.split(texts, labels):
            model = self._new_pipeline().fit(texts.iloc[train_idx], labels.iloc[train_idx])
            result[hold_idx] = model.predict_proba(texts.iloc[hold_idx])[:, 1]
        return result

    def top_terms(self, text: str, k: int = 3) -> list[str]:
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        coefficients = self.pipeline.named_steps["clf"].coef_[0]
        row = vectorizer.transform([text])
        contributions = {index: value * coefficients[index] for index, value in zip(row.indices, row.data, strict=True)}
        vocabulary = vectorizer.get_feature_names_out()
        ranked = sorted((item for item in contributions.items() if item[1] > 0), key=lambda item: -item[1])
        # Sentence punctuation is kept so a bigram glued across a full stop does not count as readable.
        readable = re.sub(r"[^a-z.!?;]+", " ", scrub(text))
        terms: list[str] = []
        for index, _ in ranked:
            term = vocabulary[index]
            # Only phrases an investigator can find in the note (stop-word removal can glue words
            # together), and none that repeats a word of a stronger term already listed.
            words = set(term.split())
            if not re.search(rf"\b{re.escape(term)}\b", readable) or any(words & set(kept.split()) for kept in terms):
                continue
            terms.append(term)
            if len(terms) == k:
                break
        return terms
