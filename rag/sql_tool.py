"""
Read-only SQL execution layer.

Security layers implemented here (L2–L3 of the guardrail stack):

* the SQLite connection is opened with ``file:...?mode=ro`` — the OS/driver
  itself refuses writes even if the validator is ever bypassed;
* a wall-clock interrupt aborts runaway queries;
* results are capped in rows and columns before they reach the LLM.

For a non-SQLite database (PostgreSQL, SQL Server), pass a DB-API connection
that was opened with a **read-only role**. Nothing else changes.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from .security import validate_sql


class SQLAnalyticsTool:
    def __init__(
        self,
        connection: Any,
        max_rows: int = 200,
        timeout_seconds: float = 10.0,
        dialect: str = "sqlite",
    ):
        self.connection = connection
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.dialect = dialect

    # ------------------------------------------------------------------ #
    @classmethod
    def from_sqlite(cls, db_path: str | Path, **kwargs) -> "SQLAnalyticsTool":
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SQLite database not found at '{path}'. "
                "Ask Member 1 for the built database or run the ETL notebook."
            )
        conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return cls(conn, **kwargs)

    # ------------------------------------------------------------------ #
    def get_schema(self) -> str:
        """Compact schema description that is injected into the SQL prompt."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        rows = cursor.fetchall()
        lines = []
        for row in rows:
            name = row[0]
            cursor.execute(f"PRAGMA table_info('{name}')")
            cols = [f"{c[1]} {c[2]}" for c in cursor.fetchall()]
            lines.append(f"{name}({', '.join(cols)})")
        return "\n".join(lines) if lines else "(schema unavailable)"

    def sample_rows(self, table: str, n: int = 3) -> list[dict]:
        result = self.execute(f"SELECT * FROM {table} LIMIT {n}")
        return result.get("rows", [])

    # ------------------------------------------------------------------ #
    def execute(self, sql: str) -> dict:
        """
        Validate then run a query.

        Always returns a dict with ``status`` in {"success", "blocked", "error"}.
        """
        check = validate_sql(sql, max_limit=self.max_rows)
        if not check["allowed"]:
            return {
                "status": "blocked",
                "reason": check["reason"],
                "sql": sql,
                "rows": [],
                "row_count": 0,
                "columns": [],
                "latency_ms": 0.0,
            }

        guarded_sql = check["sql"]
        started = time.perf_counter()
        deadline = started + self.timeout_seconds

        handler_set = False
        if isinstance(self.connection, sqlite3.Connection):
            def _interrupt():
                return 1 if time.perf_counter() > deadline else 0

            self.connection.set_progress_handler(_interrupt, 10_000)
            handler_set = True

        try:
            cursor = self.connection.cursor()
            cursor.execute(guarded_sql)
            columns = [d[0] for d in (cursor.description or [])]
            raw_rows = cursor.fetchmany(self.max_rows)
            rows = [dict(zip(columns, tuple(r))) for r in raw_rows]
            return {
                "status": "success",
                "reason": check["reason"],
                "sql": guarded_sql,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": len(rows) >= self.max_rows,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"{exc.__class__.__name__}: {exc}",
                "sql": guarded_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        finally:
            if handler_set:
                self.connection.set_progress_handler(None, 0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def format_result(result: dict, max_rows: int = 25) -> str:
        """Render a result set as a small markdown table for the LLM prompt."""
        if result["status"] != "success":
            return f"QUERY {result['status'].upper()}: {result['reason']}"
        if not result["rows"]:
            return "The query returned no rows."

        cols = result["columns"]
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = [
            "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
            for row in result["rows"][:max_rows]
        ]
        extra = (
            f"\n({result['row_count']} rows returned, showing {min(max_rows, result['row_count'])})"
            if result["row_count"] > max_rows
            else ""
        )
        return "\n".join([head, sep, *body]) + extra
