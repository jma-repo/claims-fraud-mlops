import pandas as pd
import pytest

from fraudops.rag.copilot import NOT_COVERED, Copilot, tidy
from fraudops.rag.llm import LLMUnavailable, ModelChoice, assert_local
from fraudops.rag.retrieval import CHUNK_CHARS, KnowledgeBase, bm25, chunk_markdown

DOC = """# Towing guide

## Storage yards
Ask the storage yard for an itemised invoice before paying any release fee.

## Hail claims
A policy that started days before the storm is a reason to look closer at hail damage.
"""


def knowledge_base() -> KnowledgeBase:
    rows = chunk_markdown(DOC, "towing.md")
    return KnowledgeBase(pd.DataFrame(rows), {"dimensions": 0, "chunks": len(rows)})


class FakeLLM:
    """Stands in for the local model server; returns canned answers and has no embeddings."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def embed(self, texts, kind="document"):
        raise LLMUnavailable("no embedding model")

    def chat(self, system, user, max_tokens=450, temperature=0.2):
        self.prompts.append(user)
        return self.answers.pop(0)

    def choose(self):
        return ModelChoice(chat="fake-3b", embed=None)


def test_chunking_follows_headings_and_caps_length():
    chunks = chunk_markdown(DOC + "\n## Long\n" + "word " * 600, "doc.md")
    assert [c["heading"] for c in chunks[:2]] == ["Storage yards", "Hail claims"]
    assert all(c["title"] == "Towing guide" for c in chunks)
    assert max(len(c["text"]) for c in chunks) <= CHUNK_CHARS


def test_bm25_ranks_the_matching_section_first():
    scores = bm25("release fee storage yard", ["hail damage on the roof", "storage yard release fee invoice"])
    assert scores[1] > scores[0] == 0


@pytest.mark.parametrize("base", ["http://127.0.0.1:1234/v1", "http://localhost:1234/v1", "http://host.docker.internal:1234/v1"])
def test_local_model_hosts_are_allowed(base):
    assert assert_local(base) == base


@pytest.mark.parametrize("base", ["https://api.openai.com/v1", "http://192.168.1.20:1234/v1", "file:///etc/passwd"])
def test_remote_model_hosts_are_refused(base):
    with pytest.raises(ValueError):
        assert_local(base)


def test_answer_is_generated_and_grounded_in_retrieved_text():
    llm = FakeLLM(["Ask for an itemised invoice first [1]."])
    result = Copilot(knowledge_base(), llm).ask("What should I do about a storage yard release fee?")
    assert result["mode"] == "generated" and result["retrieval"] == "keyword"
    assert result["sources"][0]["heading"] == "Storage yards"
    assert "itemised invoice" in llm.prompts[0]           # the retrieved text reached the model


def test_unrelated_questions_are_declined_without_calling_the_model():
    llm = FakeLLM([])
    result = Copilot(knowledge_base(), llm).ask("Who won the hockey game last night?")
    assert result["mode"] == "abstained" and NOT_COVERED in result["answer"]
    assert llm.prompts == []


def test_accusatory_answers_are_retried_then_withheld():
    llm = FakeLLM(["This claim is fraudulent.", "The claimant is clearly fraudulent."])
    result = Copilot(knowledge_base(), llm).ask("What should I do about a storage yard release fee?")
    assert result["mode"] == "extractive"
    assert "withheld" in result["answer"] and len(llm.prompts) == 2


def test_works_without_any_model_server():
    result = Copilot(knowledge_base(), None).ask("What should I do about a storage yard release fee?")
    assert result["mode"] == "extractive" and "[1]" in result["answer"]


def test_question_length_is_validated():
    with pytest.raises(ValueError):
        Copilot(knowledge_base(), None).ask("hi")


def test_citations_on_their_own_line_are_attached_to_the_point():
    assert tidy("- Move the vehicle\n[2]\n- Ask for the invoice\n[1], [3]") == "- Move the vehicle [2]\n- Ask for the invoice [1], [3]"
