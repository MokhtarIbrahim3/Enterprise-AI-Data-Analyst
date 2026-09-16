"""
The controlled analytics agent.

"Controlled" is the important word: the LLM never touches the database and
never produces a final number on its own. The flow is always

    question
      -> input screening (security.check_user_input)
      -> routing          (LLM router, deterministic rule fallback)
      -> tool execution   (SQL: generate -> validate -> execute
                           RAG: retrieve -> sanitize)
      -> grounded answer  (LLM writes prose over tool evidence only)
      -> audit log + structured result

Every branch returns the same result dict so Member 6's Streamlit app can
render it without special-casing.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .audit import JsonlAuditLogger
from .llm import BaseLLM, get_llm
from .prompts import (
    ANSWER_PROMPT,
    NO_TOOL_MESSAGE,
    RAG_TOOL_DESCRIPTION,
    REFUSAL_MESSAGE,
    ROUTER_PROMPT,
    SQL_GENERATION_PROMPT,
    SQL_TOOL_DESCRIPTION,
    SYSTEM_PROMPT,
)
from .retrieval import Retriever
from .security import check_user_input
from .sql_tool import SQLAnalyticsTool

# --------------------------------------------------------------------------- #
# Deterministic routing fallback
# --------------------------------------------------------------------------- #
_SQL_HINTS = (
    r"\btotal\b", r"\bsum\b", r"\bhow many\b", r"\bcount\b",
    r"\bnumber of\b", r"\btop\s+\d*\b", r"\bhighest\b", r"\blowest\b",
    r"\bbest[- ]selling\b", r"\brank\b", r"\btrend\b", r"\bcalculate\b",
    r"\bper (month|year|country|product|customer)\b", r"\bmonthly\b",
    r"\byearly\b", r"\bin \d{4}\b", r"\bfor \d{4}\b", r"\bbetween \d{4}\b",
    r"\blist\b", r"\bwhich (customers?|products?|countries)\b",
    r"\bshow me the\b", r"\brevenue (in|for|by)\b", r"\bsales (in|for|by)\b",
)

# "Strong" documentation signals: the question is explicitly about a definition,
# a convention or a column meaning.
_RAG_STRONG_HINTS = (
    r"\bdefine\b", r"\bdefinition\b", r"\bdefined\b", r"\bmean(s|ing)?\b",
    r"\bhow (is|are|do we) .*(defined|calculated|computed|measured)\b",
    r"\baccording to (our|the) (business|documentation|definition)\b",
    r"\bpolicy\b", r"\bguideline", r"\bconvention\b", r"\bdata dictionary\b",
    r"\bkpi\b", r"\bcolumn\b", r"\bdocumentation\b",
)

# "Generic" signal: a bare question form that only means RAG when nothing in the
# question asks for a computation.
_RAG_GENERIC_HINTS = (r"\bwhat (is|are|does)\b", r"\bexplain\b")


def rule_based_route(question: str) -> tuple[list[str], str]:
    q = (question or "").lower()
    sql_score = sum(bool(re.search(p, q)) for p in _SQL_HINTS)
    strong_rag = sum(bool(re.search(p, q)) for p in _RAG_STRONG_HINTS)
    generic_rag = sum(bool(re.search(p, q)) for p in _RAG_GENERIC_HINTS)

    if strong_rag and sql_score:
        return ["sql", "rag"], "Question asks for both a figure and a definition."
    if strong_rag:
        return ["rag"], "Question asks for a documented definition."
    if sql_score:
        return ["sql"], "Question asks for a calculation from the database."
    if generic_rag:
        return ["rag"], "Conceptual question with no computation requested."
    return ["rag"], "No strong signal; defaulting to documentation lookup."


# --------------------------------------------------------------------------- #
# Offline SQL templates (used only when no LLM is configured)
# --------------------------------------------------------------------------- #
# Real schema (sql/schema.sql):
#   customers(customer_id, country)
#   products(stock_code, description)
#   orders(invoice_no, customer_id, invoice_date, is_cancelled)
#   order_items(item_id, invoice_no, stock_code, quantity, unit_price, revenue)
#
# Revenue lives in order_items and must be joined through orders to reach a
# date or a customer. Cancelled orders are excluded by default everywhere —
# this mirrors the analytics_guidelines.md convention and must not be
# silently changed by a template edit.
_REVENUE_JOIN = (
    "FROM order_items oi "
    "JOIN orders o ON o.invoice_no = oi.invoice_no "
    "WHERE o.is_cancelled = 0"
)

_SQL_TEMPLATES: tuple[tuple[str, str], ...] = (
    (r"total revenue.*?(\d{4})",
     "SELECT ROUND(SUM(oi.revenue), 2) AS total_revenue " + _REVENUE_JOIN +
     " AND strftime('%Y', o.invoice_date) = '{0}'"),
    (r"(?:number of|how many) customers",
     "SELECT COUNT(DISTINCT customer_id) AS n_customers FROM customers"),
    (r"(?:number of|how many) orders.*?(\d{4})",
     "SELECT COUNT(*) AS n_orders FROM orders "
     "WHERE is_cancelled = 0 AND strftime('%Y', invoice_date) = '{0}'"),
    (r"top\s*(\d+)?\s*customers",
     "SELECT o.customer_id, ROUND(SUM(oi.revenue), 2) AS total_revenue "
     + _REVENUE_JOIN +
     " GROUP BY o.customer_id ORDER BY total_revenue DESC LIMIT {0}"),
    (r"monthly revenue|revenue by month|monthly sales",
     "SELECT strftime('%Y-%m', o.invoice_date) AS month, "
     "ROUND(SUM(oi.revenue), 2) AS total_revenue " + _REVENUE_JOIN +
     " GROUP BY month ORDER BY month"),
    (r"top\s*(\d+)?\s*(?:selling\s+)?products|highest quantity|best[- ]selling",
     "SELECT oi.stock_code, p.description, SUM(oi.quantity) AS total_quantity "
     "FROM order_items oi "
     "JOIN orders o ON o.invoice_no = oi.invoice_no "
     "JOIN products p ON p.stock_code = oi.stock_code "
     "WHERE o.is_cancelled = 0 "
     "GROUP BY oi.stock_code ORDER BY total_quantity DESC LIMIT {0}"),
)


def template_sql(question: str) -> str | None:
    q = (question or "").lower()
    for pattern, template in _SQL_TEMPLATES:
        match = re.search(pattern, q)
        if match:
            groups = [g if g else "10" for g in (match.groups() or ())]
            try:
                return template.format(*groups) if groups else template
            except IndexError:
                return template
    return None


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class AnalyticsAgent:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever | None = None,
        sql_tool: SQLAnalyticsTool | None = None,
        prediction_tool: Any = None,
        audit: Any = None,
        cache_enabled: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.llm = llm
        self.retriever = retriever
        self.sql_tool = sql_tool
        self.prediction_tool = prediction_tool
        self.audit = audit or JsonlAuditLogger()
        self.system_prompt = system_prompt
        self.cache_enabled = cache_enabled
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    @property
    def available_tools(self) -> list[str]:
        tools = []
        if self.sql_tool is not None:
            tools.append("sql")
        if self.retriever is not None:
            tools.append("rag")
        if self.prediction_tool is not None:
            tools.append("prediction")
        return tools

    # ------------------------------------------------------------------ #
    def route(self, question: str) -> dict:
        """Choose tools. LLM first, deterministic rules as fallback."""
        available = self.available_tools
        descriptions = "\n\n".join(
            d for t, d in (
                ("sql", SQL_TOOL_DESCRIPTION),
                ("rag", RAG_TOOL_DESCRIPTION),
            ) if t in available
        )
        prompt = ROUTER_PROMPT.format(
            tool_descriptions=descriptions,
            tool_names=", ".join(available),
            question=question,
        )
        raw = ""
        try:
            raw = self.llm.complete(prompt, system=self.system_prompt, max_tokens=150)
        except Exception as exc:
            raw = ""
            print(f"[agent] router LLM call failed: {exc}")

        if raw:
            try:
                payload = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(),
                                            flags=re.MULTILINE).strip())
                tools = [t for t in payload.get("tools", []) if t in available]
                if tools:
                    return {"tools": tools, "reason": payload.get("reason", ""),
                            "method": "llm"}
            except Exception:
                pass

        tools, reason = rule_based_route(question)
        tools = [t for t in tools if t in available] or available[:1]
        return {"tools": tools, "reason": reason, "method": "rules"}

    # ------------------------------------------------------------------ #
    def generate_sql(self, question: str, context: str = "") -> str:
        if self.sql_tool is None:
            return ""
        prompt = SQL_GENERATION_PROMPT.format(
            dialect=self.sql_tool.dialect,
            schema=self.sql_tool.get_schema(),
            context=context or "(none)",
            question=question,
        )
        try:
            raw = self.llm.complete(prompt, system=self.system_prompt, max_tokens=400)
        except Exception as exc:
            print(f"[agent] SQL generation failed: {exc}")
            raw = ""

        sql = re.sub(r"^```(sql)?|```$", "", (raw or "").strip(),
                     flags=re.MULTILINE).strip()
        if not sql:
            sql = template_sql(question) or ""
        return sql

    # ------------------------------------------------------------------ #
    def run(self, question: str) -> dict:
        trace_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        result: dict = {
            "answer": "",
            "tools_used": [],
            "sources": [],
            "sql": None,
            "data": [],
            "context": "",
            "status": "success",
            "trace_id": trace_id,
            "routing_reason": "",
            "security": {"input_findings": [], "document_findings": []},
            "latency_ms": 0.0,
        }

        # ---- cache -------------------------------------------------------
        key = (question or "").strip().lower()
        if self.cache_enabled and key in self._cache:
            cached = dict(self._cache[key])
            cached["cached"] = True
            return cached

        # ---- 1. input screening -----------------------------------------
        screen = check_user_input(question)
        result["security"]["input_findings"] = screen["findings"]
        if not screen["allowed"]:
            result.update(
                answer=REFUSAL_MESSAGE,
                status="blocked",
                routing_reason=screen["reason"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            self.audit.log({
                "trace_id": trace_id, "event": "blocked", "question": question,
                "status": "blocked", "reason": screen["reason"],
                "tools_used": [], "sql": None, "n_sources": 0,
                "latency_ms": result["latency_ms"],
                "injection_findings": screen["findings"],
            })
            return result

        # ---- 2. routing --------------------------------------------------
        routing = self.route(question)
        result["routing_reason"] = routing["reason"]
        tools = routing["tools"]
        self.audit.log({
            "trace_id": trace_id, "event": "route", "question": question,
            "status": "success", "tools_used": tools,
            "reason": f"{routing['method']}: {routing['reason']}",
            "sql": None, "n_sources": 0, "latency_ms": 0.0,
            "injection_findings": [],
        })

        if not tools:
            result.update(answer=NO_TOOL_MESSAGE, status="success")
            return self._finish(result, question, started)

        evidence_parts: list[str] = []

        # ---- 3a. RAG -----------------------------------------------------
        rag_context = ""
        if "rag" in tools and self.retriever is not None:
            hits = self.retriever.retrieve(question)
            findings = [f for h in hits for f in h.get("injection_findings", [])]
            result["security"]["document_findings"] = findings
            rag_context = self.retriever.format_context(hits)
            result["sources"] = self.retriever.citations(hits)
            result["context"] = rag_context
            result["tools_used"].append("rag")
            evidence_parts.append(rag_context)
            self.audit.log({
                "trace_id": trace_id, "event": "retrieval", "question": question,
                "status": "success", "tools_used": ["rag"], "sql": None,
                "reason": None, "n_sources": len(result["sources"]),
                "latency_ms": 0.0, "injection_findings": findings,
            })

        # ---- 3b. SQL -----------------------------------------------------
        if "sql" in tools and self.sql_tool is not None:
            # recorded as "used" as soon as it is selected, so tool-selection
            # accuracy measures routing rather than execution success
            result["tools_used"].append("sql")
            sql = self.generate_sql(question, context=rag_context)
            if not sql:
                result["status"] = "error"
                evidence_parts.append(
                    "SQL EVIDENCE: the system could not produce a query for this "
                    "question (no LLM configured and no matching template)."
                )
            else:
                sql_result = self.sql_tool.execute(sql)
                result["sql"] = sql_result["sql"]
                result["data"] = sql_result["rows"]
                if sql_result["status"] != "success":
                    result["status"] = sql_result["status"]
                evidence_parts.append(
                    "SQL EVIDENCE\nquery:\n" + sql_result["sql"] + "\nresult:\n"
                    + SQLAnalyticsTool.format_result(sql_result)
                )
                self.audit.log({
                    "trace_id": trace_id, "event": "sql", "question": question,
                    "status": sql_result["status"], "tools_used": ["sql"],
                    "sql": sql_result["sql"], "reason": sql_result["reason"],
                    "n_sources": 0, "latency_ms": sql_result["latency_ms"],
                    "injection_findings": [],
                })

        # ---- 4. grounded answer -----------------------------------------
        evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"
        answer = ""
        try:
            answer = self.llm.complete(
                ANSWER_PROMPT.format(evidence=evidence, question=question),
                system=self.system_prompt,
                max_tokens=700,
            )
        except Exception as exc:
            print(f"[agent] answer generation failed: {exc}")

        result["answer"] = answer.strip() or self._fallback_answer(result, evidence)
        return self._finish(result, question, started)

    # ------------------------------------------------------------------ #
    def _fallback_answer(self, result: dict, evidence: str) -> str:
        """Deterministic answer used when no LLM is configured."""
        lines = ["(Deterministic mode — no LLM configured; showing raw evidence.)"]
        if result.get("data"):
            lines.append("\nQuery result:")
            lines.append(json.dumps(result["data"][:10], indent=2, default=str))
        if result.get("sources"):
            lines.append("\nRelevant documentation: " + ", ".join(result["sources"]))
            lines.append(evidence[:1200])
        return "\n".join(lines)

    def _finish(self, result: dict, question: str, started: float) -> dict:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["cached"] = False
        self.audit.log({
            "trace_id": result["trace_id"], "event": "answer", "question": question,
            "status": result["status"], "tools_used": result["tools_used"],
            "sql": result["sql"], "reason": result["routing_reason"],
            "n_sources": len(result["sources"]), "latency_ms": result["latency_ms"],
            "injection_findings": result["security"]["document_findings"],
        })
        if self.cache_enabled and result["status"] == "success":
            self._cache[(question or "").strip().lower()] = result
        return result


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_agent(
    knowledge_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    index_path: str | Path | None = None,
    rebuild_index: bool = False,
    llm: BaseLLM | None = None,
    audit: Any = None,
    prediction_tool: Any = None,
) -> AnalyticsAgent:
    """
    One-call construction used by the notebook, the tests and the Streamlit app.

    Reads defaults from ``src.utils.config`` when available, otherwise from
    environment variables, otherwise from repo-relative defaults.
    """
    try:
        from src.utils import config as cfg  # type: ignore
    except Exception:
        cfg = None

    def setting(name: str, default):
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
        import os

        return os.getenv(name, default)

    knowledge_dir = knowledge_dir or setting("KNOWLEDGE_DIR", "data")
    db_path = db_path or setting("DB_PATH", "data/retail.db")
    index_path = index_path or setting("VECTOR_DB_PATH", "models/vector_index")

    retriever = None
    try:
        retriever = Retriever.from_knowledge_base(
            knowledge_dir, index_path=index_path, rebuild=rebuild_index
        )
    except Exception as exc:
        print(f"[build_agent] RAG disabled: {exc}")

    sql_tool = None
    try:
        sql_tool = SQLAnalyticsTool.from_sqlite(db_path)
    except Exception as exc:
        print(f"[build_agent] SQL tool disabled: {exc}")

    return AnalyticsAgent(
        llm=llm or get_llm(),
        retriever=retriever,
        sql_tool=sql_tool,
        prediction_tool=prediction_tool,
        audit=audit,
    )
