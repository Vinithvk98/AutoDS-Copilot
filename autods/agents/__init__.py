"""LangGraph multi-agent orchestration layer.

The pipeline in autods/pipeline does the real ML work. This package wraps each
stage as a node in a stateful LangGraph graph, with human-in-the-loop interrupts
at the two decision points from the original sketch: approving the cleaning plan
and choosing the model.
"""
from .graph import build_graph, AutoDSState  # noqa: F401
