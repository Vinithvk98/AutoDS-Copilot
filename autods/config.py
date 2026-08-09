"""Central configuration for AutoDS Copilot."""
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# EDA / modelling defaults
MAX_CATEGORICAL_CARDINALITY = 20   # above this, a column is treated as high-cardinality
CLASSIFICATION_MAX_UNIQUE = 20     # numeric target with <= this many unique values -> classification
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optional LLM narration. If an OpenAI-compatible key is present the insight
# modules produce richer prose; otherwise they fall back to solid rule-based text.
import os
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
LLM_MODEL = os.getenv("AUTODS_LLM_MODEL", "gpt-4o-mini")
