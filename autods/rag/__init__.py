"""The retrieval layer: knowledge base + run memory, with citations, experience
retrieval, and grounded answering."""
from .store import (Retriever, get_retriever, record_run, rationale, answer,  # noqa: F401
                    fingerprint, similar_runs, build_run_docs)
