"""
Shared configuration (owned jointly by Member 5 and Member 6).

Everything is read from environment variables with repo-relative defaults, so
the project runs identically on a laptop, in CI and inside Docker.

NEVER commit real secrets. `.env` is gitignored; `.env.example` is the template.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional dependency, but recommended
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_name: str, default: str) -> str:
    value = os.getenv(env_name, default)
    p = Path(value)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


DATA_DIR = _path("DATA_DIR", "data")
KNOWLEDGE_DIR = _path("KNOWLEDGE_DIR", "data")
MODELS_DIR = _path("MODELS_DIR", "models")
REPORTS_DIR = _path("REPORTS_DIR", "reports")
SQL_DIR = _path("SQL_DIR", "sql")

DB_PATH = _path("DB_PATH", "data/retail.db")
VECTOR_DB_PATH = _path("VECTOR_DB_PATH", "models/vector_index")
AUDIT_LOG_PATH = _path("AUDIT_LOG_PATH", "reports/agent_audit.jsonl")

# --------------------------------------------------------------------------- #
# LLM / embeddings
# --------------------------------------------------------------------------- #
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "rule")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------------------------------- #
# Agent limits (guardrails)
# --------------------------------------------------------------------------- #
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))
SQL_TIMEOUT_SECONDS = float(os.getenv("SQL_TIMEOUT_SECONDS", "10"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.15"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))


def summary() -> dict:
    """Safe-to-print configuration (never exposes the API key)."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DB_PATH": DB_PATH,
        "KNOWLEDGE_DIR": KNOWLEDGE_DIR,
        "VECTOR_DB_PATH": VECTOR_DB_PATH,
        "LLM_PROVIDER": LLM_PROVIDER,
        "LLM_MODEL": LLM_MODEL,
        "LLM_API_KEY": "set" if LLM_API_KEY else "not set",
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "SQL_MAX_ROWS": SQL_MAX_ROWS,
        "RETRIEVAL_TOP_K": RETRIEVAL_TOP_K,
    }
