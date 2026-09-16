
"""
rag/ — Knowledge layer + controlled analytics agent (Member 5).

Public API used by the Streamlit app (Member 6) and by the notebooks:

    from rag import build_agent

    agent = build_agent()
    result = agent.run("What was total revenue in 2011?")

`result` is always a dict with the same keys (see rag.agent.AgentResult).
"""

from .agent import AnalyticsAgent, build_agent  # noqa: F401
from .retrieval import Retriever  # noqa: F401
from .vector_store import VectorStore  # noqa: F401
from .sql_tool import SQLAnalyticsTool  # noqa: F401
from .security import validate_sql, scan_for_injection, sanitize_document  # noqa: F401

__all__ = [
    "AnalyticsAgent",
    "build_agent",
    "Retriever",
    "VectorStore",
    "SQLAnalyticsTool",
    "validate_sql",
    "scan_for_injection",
    "sanitize_document",
]

__version__ = "0.1.0"
