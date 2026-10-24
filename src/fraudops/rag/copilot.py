"""Investigator copilot: answers grounded in policy documents, this run's briefs and the claim file.

Three guardrails sit outside the prompt, because a 3B local model cannot be trusted to follow
every instruction:
  * abstain without calling the model when retrieval finds nothing relevant;
  * replace any answer that calls a claim fraudulent or recommends a denial;
  * keep working with no model server at all (extractive answer from the same sources).
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fraudops.rag.llm import LLMUnavailable, LocalLLM
from fraudops.rag.retrieval import Hit, KnowledgeBase

SYSTEM_PROMPT = """You are the SIU triage copilot for a Canadian auto insurer. Answer the question from the material provided, in English, in at most 5 sentences or bullet points. Put the label of the source after each point, like [1] or [C].
- Facts about the claim come only from [C]; never say the claim shows an indicator unless [C] states it.
- A model score is a triage signal, not evidence: never call a claim or a person fraudulent and never recommend denying a claim; recommend verification steps instead.
- Only if no part of the material is about the question, reply exactly: The material does not cover this."""
# Claim questions: the "why" is stated from the reason codes, word for word, and the model only
# writes the next steps. A 3B model asked for both tends to borrow indicators from the guide
# and attribute them to the claim.
CLAIM_TASK = ("\n\nWrite only the first verification steps for this claim, as at most 4 bullet points taken from the "
              "numbered sources, each ending with its label. Do not repeat why the claim was flagged.")
NOT_COVERED = "The material does not cover this."
RETRY_NOTE = "\n\nYour previous answer judged the claim. Rewrite it as evidence and next steps only, without the words fraudulent or deny."
# Abstain gate, set from eval/copilot_eval.yaml: unrelated questions ("who won the Stanley Cup")
# score below these; insurance-adjacent ones ("vacation days for investigators") do not, and are
# left to the model's instruction to decline. Raw signals, not the per-query normalised score.
MIN_SEMANTIC = 0.60   # best cosine similarity, hybrid mode
MIN_COVERAGE = 0.34   # share of the question's content words in the top passage, keyword-only mode
ACCUSATORY = re.compile(r"\b(is|are|was|were|as|be|seems?|appears?)\s+(clearly\s+|likely\s+|probably\s+|potentially\s+|possibly\s+)?"
                        r"(fraudulent|a fraud|frauds|fraudsters)\b|\bden(y|ying|ial of)\s+(the|this|that)\s+claim\b", re.IGNORECASE)

# Reason features -> words that pull the matching guidance out of the knowledge base.
REASON_QUERIES = {
    "rep_clinic_prior_claims_365d": "same representative and clinic staged collision ring",
    "clinic_prior_claims_90d": "clinic treatment plans injury build-up",
    "phone_prior_claims_365d": "phone numbers shared with other claims network review",
    "address_prior_claims_365d": "addresses shared with other claims network review",
    "n_injured": "occupants injured soft-tissue staged collision statements",
    "injured_share": "every occupant claims injury phantom passengers",
    "soft_tissue_only": "soft-tissue injuries neck and back pain",
    "policy_tenure_days": "policy only weeks old post-loss inception",
    "report_delay_days": "injury first reported weeks after the loss",
    "shop_avg_repair_ratio_180d": "repair build-up body shop estimates above peers",
    "repair_to_value": "repair estimate high compared with vehicle value supplementary estimate",
    "log_repair_estimate": "repair build-up supplementary estimate",
    "tow_fee": "towing and storage scheme release fee storage yard",
    "storage_days": "storage charges accruing release fee",
    "tow_distance_km": "towed to a distant storage yard",
    "tow_avg_storage_days_180d": "tow operator storage scheme",
    "narrative_risk": "red-flag indicators in adjuster notes",
    "loss_type": "red-flag indicators",
    "vehicle_value": "owner give-up theft high-value vehicle",
    "total_loss": "owner give-up theft",
}


def _money(value: float) -> str:
    return f"${value:,.0f}"


def tidy(answer: str) -> str:
    """Small models often put a citation on a line of its own; attach it to the line it cites."""
    return re.sub(r"\n[ \t]*((?:\[(?:\d+|C)\][, \t]*)+)(?=\n|$)", lambda m: " " + m.group(1).strip(), answer).strip()


class ClaimDirectory:
    """Scored claims plus their ring and emerging-pattern membership, for the claim-file context."""

    def __init__(self, scored: pd.DataFrame, rings: dict[str, Any], segments: dict[str, Any],
                 titles: dict[str, dict[str, str]] | None = None):
        self.scored = scored.set_index("claim_id", drop=False)
        titles = titles or {}
        self.membership: dict[str, list[str]] = {}
        for ring in rings.get("rings", [])[:5]:
            title = titles.get(ring["id"], {}).get("title", "")
            label = (f"part of ring {ring['id']} ({title}): {ring['claims']} claims linked through "
                     f"{ring['shared_phones']} shared phone numbers and {ring['shared_addresses']} shared addresses")
            for claim_id in ring["claim_ids"]:
                self.membership.setdefault(claim_id, []).append(label)
        for pattern in segments.get("emerging", []):
            title = titles.get(pattern["id"], {}).get("title", "")
            label = (f"member of emerging pattern {pattern['id']} ({title}): {pattern['claims']} similar claims this month, "
                     "a kind of claim the model has not been trained on")
            for claim_id in pattern["claim_ids"]:
                self.membership.setdefault(claim_id, []).append(label)

    @classmethod
    def from_artifacts(cls, scored_path: Path, reports: Path) -> ClaimDirectory:
        def read(name: str) -> dict[str, Any]:
            path = reports / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        return cls(pd.read_parquet(scored_path), read("rings.json"), read("segments.json"), read("pattern_titles.json"))

    def get(self, claim_id: str) -> pd.Series | None:
        return self.scored.loc[claim_id] if claim_id in self.scored.index else None

    def claim_file(self, claim_id: str) -> tuple[str, list[dict[str, Any]]] | None:
        row = self.get(claim_id)
        if row is None:
            return None
        reasons_for = json.loads(row["reasons_for"])
        reasons_against = json.loads(row["reasons_against"])
        injury, repair = math.expm1(row["log_injury_reserve"]), math.expm1(row["log_repair_estimate"])
        providers = ", ".join(f"{label} {row[column]}" for label, column in
                              [("body shop", "body_shop_id"), ("clinic", "clinic_id"), ("tow operator", "tow_company_id"),
                               ("representative", "rep_id")] if pd.notna(row[column])) or "none recorded"
        links = self.membership.get(claim_id) or ["no shared phone numbers, addresses or emerging pattern"]
        text = "\n".join([
            f"Claim file {claim_id} (synthetic data)",
            f"Loss: {str(row['loss_type']).replace('_', ' ')} in {row['city']} ({row['province']}) at "
            f"{int(row['loss_hour']):02d}:00, reported {pd.Timestamp(row['report_ts']):%Y-%m-%d}, "
            f"{int(row['report_delay_days'])} days after the loss.",
            f"Amounts: total claimed {_money(row['total_claimed'])} (injury reserve {_money(injury)}, repair estimate "
            f"{_money(repair)}, towing and storage {_money(row['tow_fee'])}, {int(row['storage_days'])} storage days, "
            f"towed {row['tow_distance_km']:.0f} km).",
            f"People: {int(row['n_occupants'])} occupants, {int(row['n_injured'])} claiming injury, soft-tissue only: "
            f"{'yes' if row['soft_tissue_only'] else 'no'}; police report: {'yes' if row['police_report'] else 'no'}; "
            f"independent witnesses: {int(row['witnesses'])}.",
            f"Policy: {int(row['policy_tenure_days'])} days old at the loss; {int(row['prior_claims_3y'])} prior claims in 3 "
            f"years. Providers: {providers}.",
            f"Triage model: fraud probability {row['probability']:.0%}, expected net saving of investigating "
            f"{_money(row['expected_value'])}, band {row['band']}, queue rank {int(row['queue_rank'])}.",
            "Raising the score: " + ("; ".join(item["text"] for item in reasons_for) or "nothing notable") + ".",
            "Lowering the score: " + ("; ".join(item["text"] for item in reasons_against) or "nothing notable") + ".",
            "Links: " + "; ".join(links) + ".",
            f"Adjuster notes: \"{row['narrative']}\"",
        ])
        return text, reasons_for


class Copilot:
    def __init__(self, kb: KnowledgeBase, llm: LocalLLM | None, claims: ClaimDirectory | None = None,
                 generate: bool = True, k: int = 4):
        self.kb, self.llm, self.claims, self.generate, self.k = kb, llm, claims, generate, k

    def ask(self, question: str, claim_id: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        question = (question or "").strip()
        if not 3 <= len(question) <= 500:
            raise ValueError("A question must be between 3 and 500 characters")

        claim_text, query = None, question
        if claim_id:
            if self.claims is None or (found := self.claims.claim_file(claim_id)) is None:
                raise KeyError(f"Unknown claim {claim_id}")
            claim_text, reasons = found
            hints = [REASON_QUERIES.get(item["feature"], "") for item in reasons[:3]]
            if "emerging pattern" in claim_text:
                hints.append("towing and storage schemes new pattern")
            if "part of ring" in claim_text:
                hints.append("provider and network review ring")
            query = question + " " + " ".join(h for h in hints if h)

        hits, retrieval, evidence = self.kb.search(query, self.llm, k=self.k)
        sources = [hit.as_dict() for hit in hits]
        if claim_text:
            sources.insert(0, {"label": "C", "source": f"claim file {claim_id}", "title": "Claim file",
                               "heading": claim_id, "score": None, "semantic": None, "lexical": None,
                               "snippet": claim_text[:280]})
        relevant = bool(hits) and (evidence["semantic_max"] >= MIN_SEMANTIC if retrieval == "hybrid"
                                   else evidence["coverage"] >= MIN_COVERAGE)

        if not claim_text and not relevant:
            mode, answer, model = "abstained", (f"{NOT_COVERED} None of the policy documents or this run's briefs "
                                                 "matched the question closely enough to answer it."), None
        else:
            mode, answer, model = self._answer(question, hits, claim_text)
        return {
            "question": question,
            "claim_id": claim_id,
            "answer": answer,
            "mode": mode,
            "retrieval": retrieval,
            "sources": sources,
            "model": model,
            "local_only": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    def _answer(self, question: str, hits: list[Hit], claim_text: str | None) -> tuple[str, str, str | None]:
        material = "\n\n".join(f"[{hit.label}] {hit.title} - {hit.heading}\n{hit.text}" for hit in hits)
        preface = ""
        if claim_text:
            material = f"[C] {claim_text}\n\n{material}"
            facts = dict(line.split(": ", 1) for line in claim_text.splitlines()[1:] if ": " in line)
            reasons = [r for r in facts.get("Raising the score", "").rstrip(".").split("; ") if r]
            preface = ("Why it is in the queue [C]:\n" + "".join(f"- {r}\n" for r in reasons)
                       + f"- Triage model: {facts.get('Triage model', '').rstrip('.')}\n"
                       + f"- Links: {facts.get('Links', '').rstrip('.')}\n\nWhat to check first:\n")
        if self.generate and self.llm is not None:
            prompt = f"[Material]\n{material}\n\n[Question]\n{question}" + (CLAIM_TASK if claim_text else "")
            try:
                for attempt in (prompt, prompt + RETRY_NOTE):
                    answer = tidy(self.llm.chat(SYSTEM_PROMPT, attempt, temperature=0.1))
                    if not ACCUSATORY.search(answer):
                        return "generated", preface + answer, self.llm.choose().chat
                # Twice in a row the model judged the claim: show the evidence instead, and say why.
                return "extractive", self._extractive(hits, claim_text, note="The generated answer was withheld because it "
                                                      "judged the claim instead of listing evidence."), None
            except LLMUnavailable:
                pass
        return "extractive", self._extractive(hits, claim_text), None

    @staticmethod
    def _extractive(hits: list[Hit], claim_text: str | None, note: str | None = None) -> str:
        lines = [note or "Local model offline: showing the most relevant evidence instead of a written answer."]
        if claim_text:
            reasons = next((line for line in claim_text.splitlines() if line.startswith("Raising the score")), "")
            lines.append(f"- {reasons} [C]")
        for hit in hits[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", hit.text.replace("\n", " "))
            lines.append(f"- {hit.heading}: {' '.join(sentences[:2])} [{hit.label}]")
        return "\n".join(lines)
