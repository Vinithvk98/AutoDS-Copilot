"""Modular, framework-free pipeline stages.

Each stage is a plain function so it can be unit-tested on its own and also
wrapped as a node in the LangGraph agent layer (see autods/agents/graph.py).
"""
from . import (loader, preclean, task_detect, eda, preprocess, model_select,  # noqa: F401
               train, evaluate, insights, leaderboard)
