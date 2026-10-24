"""Knowledge base: markdown chunking, a DuckDB vector index and hybrid (vector + BM25) search.

Chunking, tokenisation and the 0.55/0.45 hybrid weighting are ported from the local-rag module
in secure-dev-platform, where keyword scoring fixed questions that name a specific term the
English-first embedding model ranks poorly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fraudops.rag.llm import LLMUnavailable, LocalLLM

CHUNK_CHARS = 900
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45
CJK_RUN = f"[{chr(0x3400)}-{chr(0x9FFF)}]+"  # built from code points to keep this file ASCII


def chunk_markdown(text: str, source: str) -> list[dict[str, str]]:
    """Split on headings, then cap chunk length so one long section cannot dominate retrieval."""
    title = source
    heading = "Introduction"
    chunks: list[dict[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if len(body) > 40:
            chunks.append({"source": source, "title": title, "heading": heading, "text": body})
        buffer.clear()

    for line in text.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if match:
            flush()
            if len(match.group(1)) == 1:
                title = match.group(2).strip()
            heading = match.group(2).strip()
            continue
        if len(line) > CHUNK_CHARS:
            for start in range(0, len(line), CHUNK_CHARS):
                buffer.append(line[start:start + CHUNK_CHARS])
                flush()
            continue
        if buffer and len("\n".join(buffer)) + len(line) + 1 > CHUNK_CHARS:
            flush()
        buffer.append(line)
    flush()
    return chunks


def tokenize(text: str) -> list[str]:
    """Latin words plus CJK unigrams and bigrams, so a question in Chinese still gets keyword hits."""
    value = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9._+-]*", value)
    for run in re.findall(CJK_RUN, value):
        for index, char in enumerate(run):
            tokens.append(char)
            if index + 1 < len(run):
                tokens.append(run[index:index + 2])
    return tokens


def bm25(query: str, documents: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    query_tokens = set(tokenize(query))
    tokenized = [tokenize(document) for document in documents]
    average = sum(map(len, tokenized)) / max(len(tokenized), 1)
    frequency: Counter[str] = Counter()
    for tokens in tokenized:
        frequency.update(set(tokens))
    total = max(len(tokenized), 1)
    scores = np.zeros(len(documents))
    for position, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            count = counts.get(token, 0)
            if not count:
                continue
            idf = math.log(1 + (total - frequency[token] + 0.5) / (frequency[token] + 0.5))
            score += idf * count * (k1 + 1) / (count + k1 * (1 - b + b * len(tokens) / (average or 1)))
        scores[position] = score
    return scores


STOPWORDS = set("a about an and any are as at be can could do does for from how i if in is it its me my of on or our "
                "should so that the their there this to was we what when where which who why will with would you your".split())


def coverage(query: str, document: str) -> float:
    """Share of the question's content words found in a passage. Unlike a raw BM25 score it does not
    depend on corpus size, so it works as an abstain signal when there are no embeddings."""
    words = {token for token in tokenize(query) if token not in STOPWORDS and len(token) > 1}
    if not words:
        return 0.0
    present = set(tokenize(document))
    return len(words & present) / len(words)


@dataclass
class Hit:
    label: str
    source: str
    title: str
    heading: str
    text: str
    score: float
    semantic: float | None
    lexical: float

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "source": self.source, "title": self.title, "heading": self.heading,
                "score": round(self.score, 3), "semantic": None if self.semantic is None else round(self.semantic, 3),
                "lexical": round(self.lexical, 3), "snippet": self.text[:280]}


class KnowledgeBase:
    def __init__(self, chunks: pd.DataFrame, meta: dict[str, Any]):
        self.chunks = chunks.reset_index(drop=True)
        self.meta = meta
        self._con: duckdb.DuckDBPyConnection | None = None
        self.dimensions = int(meta.get("dimensions") or 0)
        if self.dimensions:
            # The vector index lives in DuckDB: array_cosine_similarity over a FLOAT[n] column.
            # At a few hundred chunks a scan is instant; past ~100k, swap in a vector database.
            self._con = duckdb.connect()
            self._con.register("chunks_frame", self.chunks[["embedding"]].rename_axis("row_id").reset_index())
            self._con.execute(f"CREATE TABLE kb AS SELECT row_id, embedding::FLOAT[{self.dimensions}] AS embedding "
                              "FROM chunks_frame")
            self._con.unregister("chunks_frame")

    @property
    def has_vectors(self) -> bool:
        return self._con is not None

    # --- build / persist ----------------------------------------------------------------------
    @classmethod
    def build(cls, documents: list[Path], llm: LocalLLM | None) -> KnowledgeBase:
        rows = []
        for path in documents:
            rows += chunk_markdown(path.read_text(encoding="utf-8"), path.name)
        if not rows:
            raise ValueError("No knowledge documents found")
        chunks = pd.DataFrame(rows)
        meta: dict[str, Any] = {"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "documents": len(documents),
                                "chunks": len(chunks), "embed_model": None, "dimensions": 0}
        if llm is not None:
            try:
                vectors = llm.embed([f"{r.title} - {r.heading}\n{r.text}" for r in chunks.itertuples()], kind="document")
                chunks["embedding"] = [vector.round(6).tolist() for vector in vectors]
                meta.update(embed_model=llm.choose().embed, dimensions=int(vectors.shape[1]))
            except LLMUnavailable as error:
                meta["embedding_skipped"] = str(error)
        return cls(chunks, meta)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KnowledgeBase:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        chunks = pd.read_parquet(path)
        if "embedding" not in chunks:
            meta["dimensions"] = 0
        return cls(chunks, meta)

    # --- search -------------------------------------------------------------------------------
    def search(self, query: str, llm: LocalLLM | None = None, k: int = 4) -> tuple[list[Hit], str, dict[str, float | None]]:
        documents = [f"{r.title} {r.heading}\n{r.text}" for r in self.chunks.itertuples()]
        raw_lexical = bm25(query, documents)
        # Normalised per query, so the hybrid score ranks well but cannot tell a good match from
        # the best of a bad lot; `evidence` keeps the raw strengths for the abstain decision.
        lexical = raw_lexical / raw_lexical.max() if raw_lexical.max() > 0 else raw_lexical
        semantic = None
        mode = "keyword"
        if self.has_vectors and llm is not None:
            try:
                vector = llm.embed([query], kind="query")[0]
                cursor = self._con.cursor()  # one cursor per search: the API answers questions concurrently
                try:
                    rows = cursor.execute(
                        f"SELECT row_id, array_cosine_similarity(embedding, ?::FLOAT[{self.dimensions}]) AS similarity FROM kb",
                        [vector.tolist()],
                    ).fetchall()
                finally:
                    cursor.close()
                semantic = np.zeros(len(self.chunks))
                for row_id, similarity in rows:
                    semantic[row_id] = similarity
                mode = "hybrid"
            except LLMUnavailable:
                semantic = None
        score = SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical if semantic is not None else lexical
        order = np.argsort(-score, kind="stable")[:k]
        hits = []
        for rank, index in enumerate(order, start=1):
            row = self.chunks.iloc[index]
            hits.append(Hit(label=str(rank), source=row["source"], title=row["title"], heading=row["heading"],
                            text=row["text"], score=float(score[index]),
                            semantic=None if semantic is None else float(semantic[index]), lexical=float(lexical[index])))
        evidence = {"semantic_max": None if semantic is None else float(semantic.max()), "bm25_max": float(raw_lexical.max()),
                    "coverage": coverage(query, documents[order[0]]) if len(order) else 0.0}
        return hits, mode, evidence
