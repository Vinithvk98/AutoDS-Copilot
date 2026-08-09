"""The retrieval layer. Lexical TF-IDF, offline, no API key."""
from autods.rag import store, knowledge


def test_knowledge_base_is_populated():
    assert len(knowledge.KNOWLEDGE_BASE) >= 20


def test_retriever_finds_relevant_guidance():
    r = store.get_retriever(include_memory=False)
    hits = r.query("how should I handle missing values", k=3)
    assert hits, "expected at least one retrieved chunk"
    joined = " ".join(str(h).lower() for h in hits)
    assert "missing" in joined or "impute" in joined


def test_fingerprint_is_numeric_vector(prof, task):
    fp = store.fingerprint(prof, task)
    assert isinstance(fp, dict) and len(fp) >= 3


def test_answer_returns_grounded_dict():
    res = store.answer("why can accuracy mislead on imbalanced data")
    assert isinstance(res, dict)
    # some textual answer content is present
    assert any(isinstance(v, str) and v.strip() for v in res.values())
