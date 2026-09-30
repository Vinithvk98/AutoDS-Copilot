"""Central configuration for AutoDS Copilot."""
import os
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines from a local .env into the environment, without
    overriding anything already set. Dependency-free, so no python-dotenv needed.
    The .env file is gitignored, so keys never get committed."""
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except FileNotFoundError:
        pass
    except Exception:
        pass


_load_dotenv(ROOT / ".env")

# EDA / modelling defaults
MAX_CATEGORICAL_CARDINALITY = 20   # above this, a column is treated as high-cardinality
CLASSIFICATION_MAX_UNIQUE = 20     # numeric target with <= this many unique values -> classification
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Optional LLM layer -------------------------------------------------------
# The copilot's generated prose (grounded answers, insights, recommendation
# rationale) is produced by autods/llm.py when a provider is configured, and
# falls back to rule-based text otherwise. Provider is auto-detected from the
# keys present, or forced with AUTODS_LLM_PROVIDER. Keys live in the environment
# (see .env.example) and are never committed. Use autods.llm.available() to test
# whether a model is wired in.
LLM_PROVIDER = os.getenv("AUTODS_LLM_PROVIDER", "auto")   # auto|openai|anthropic|ollama|none
LLM_MODEL = os.getenv("AUTODS_LLM_MODEL", "")             # blank -> per-provider default
LLM_TEMPERATURE = float(os.getenv("AUTODS_LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("AUTODS_LLM_MAX_TOKENS", "700"))
