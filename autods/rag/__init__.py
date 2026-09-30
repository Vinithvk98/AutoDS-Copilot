"""The retrieval layer: knowledge base + run memory, with citations, experience
retrieval, and grounded answering."""
from .store import (Retriever, get_retriever, record_run, rationale, answer,  # noqa: F401
                    retrieve_for_answer, fingerprint, similar_runs, build_run_docs)
from .generate import answer_stream  # noqa: F401
