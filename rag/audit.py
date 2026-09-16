
"""
Audit log for agent actions (required deliverable: "Audit logs for agent actions").

Contract agreed with Member 6 (MLOps / Streamlit):
    the agent calls ``audit_logger.log(event: dict)``.
Any object with a ``.log(dict)`` method can be injected here — if Member 6's
``src/utils/logger.py`` exposes one, pass it to ``build_agent(audit=...)`` and
this default implementation is not used.

Event schema (stable — Member 6 can parse it):
    {
      "timestamp": ISO-8601 UTC,
      "trace_id": str,
      "event": "question" | "route" | "sql" | "retrieval" | "answer" | "blocked",
      "question": str | None,
      "tools_used": [str],
      "sql": str | None,
      "status": "success" | "blocked" | "error",
      "reason": str | None,
      "latency_ms": float,
      "n_sources": int,
      "injection_findings": [ ... ]
    }
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class JsonlAuditLogger:
    """Append-only JSONL log. Safe for Streamlit's threaded reruns."""

    def __init__(self, path: str | Path = "reports/agent_audit.jsonl", echo: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()
        self.events: list[dict] = []

    def log(self, event: dict) -> dict:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self._lock:
            self.events.append(record)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.echo:
            print(f"[audit] {record.get('event')} :: {record.get('status')}")
        return record

    def tail(self, n: int = 20) -> list[dict]:
        return self.events[-n:]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.events)


class NullAuditLogger:
    def log(self, event: dict) -> dict:
        return event

    def tail(self, n: int = 20) -> list[dict]:
        return []
