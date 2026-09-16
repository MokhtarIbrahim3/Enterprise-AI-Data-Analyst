"""
Guardrails: read-only SQL enforcement + prompt-injection defence.

This module is deliberately the smallest, most-tested file in the package —
it is the thing that stops "Do not permit unrestricted LLM-generated SQL
against a production database" from becoming a finding in the defence.

Defence in depth (all four layers are required, none is sufficient alone):

  L1  This validator                     — syntactic allow-list.
  L2  Read-only DB connection            — see sql_tool.py (`mode=ro`).
  L3  Row/time limits                    — see sql_tool.py.
  L4  Audit log of every attempt         — see audit.py.
"""

from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# SQL validation
# --------------------------------------------------------------------------- #
FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "REPLACE", "MERGE", "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA",
    "VACUUM", "REINDEX", "EXEC", "EXECUTE", "CALL", "COPY", "LOAD_FILE",
    "OUTFILE", "DUMPFILE", "SHUTDOWN", "SET",
)

ALLOWED_STARTS: tuple[str, ...] = ("SELECT", "WITH")

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")


def strip_sql_noise(sql: str) -> str:
    """Remove comments and string literals so keyword scanning can't be fooled."""
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    sql = _STRING_LITERAL.sub(" '' ", sql)
    return re.sub(r"\s+", " ", sql).strip()


def validate_sql(
    sql: str,
    max_limit: int = 500,
    extra_forbidden: Iterable[str] = (),
) -> dict:
    """
    Validate an LLM-generated query.

    Returns
    -------
    dict with keys: ``allowed`` (bool), ``reason`` (str), ``sql`` (the query,
    with a LIMIT appended when it was missing).
    """
    if not sql or not sql.strip():
        return {"allowed": False, "reason": "Empty query.", "sql": sql}

    raw = sql.strip().rstrip(";").strip()

    # 0. no SQL comments at all. Generated analytical SQL never needs them and
    #    they are the classic way to smuggle a payload past a keyword scanner.
    if _LINE_COMMENT.search(raw) or _BLOCK_COMMENT.search(raw):
        return {
            "allowed": False,
            "reason": "SQL comments are not allowed in generated queries.",
            "sql": raw,
        }

    scan = strip_sql_noise(raw).upper()

    # 1. single statement only (stacked-query defence)
    body = strip_sql_noise(raw)
    if len([s for s in body.split(";") if s.strip()]) > 1:
        return {
            "allowed": False,
            "reason": "Multiple SQL statements are not allowed.",
            "sql": raw,
        }

    # 2. must be a read
    if not scan.startswith(ALLOWED_STARTS):
        return {
            "allowed": False,
            "reason": "Only SELECT / WITH (read-only) queries are allowed.",
            "sql": raw,
        }

    # 3. no write or admin keywords anywhere
    for keyword in tuple(FORBIDDEN_KEYWORDS) + tuple(extra_forbidden):
        if re.search(rf"\b{re.escape(keyword)}\b", scan):
            return {
                "allowed": False,
                "reason": f"Forbidden SQL operation detected: {keyword}.",
                "sql": raw,
            }

    # 4. a WITH block must still end in a SELECT
    if scan.startswith("WITH") and " SELECT " not in f" {scan} ":
        return {
            "allowed": False,
            "reason": "WITH clause does not resolve to a SELECT.",
            "sql": raw,
        }

    # 5. always bound the result set
    guarded = raw
    if not re.search(r"\bLIMIT\b", scan):
        guarded = f"{raw}\nLIMIT {max_limit}"

    return {"allowed": True, "reason": "Query passed read-only validation.", "sql": guarded}


# --------------------------------------------------------------------------- #
# Prompt-injection defence
# --------------------------------------------------------------------------- #
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore (all|any|the)?\s*(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
     "instruction_override"),
    (r"disregard (all|any|the)?\s*(previous|prior|above)\s+(instructions|rules)",
     "instruction_override"),
    (r"forget (everything|all|your) (you|instructions|rules|training)",
     "instruction_override"),
    (r"you are now (a|an|the)\b", "role_hijack"),
    (r"\bnew (system )?(instructions|prompt)\b", "role_hijack"),
    (r"\b(reveal|show|print|output|leak|expose|repeat)\b.{0,30}\b"
     r"(system prompt|api[_ ]?key|password|secret|credential|token|\.env)\b",
     "secret_exfiltration"),
    (r"\b(drop|delete|update|truncate|alter)\b\s+\b(table|from|database)\b",
     "destructive_sql"),
    (r"\bdo not (tell|mention|cite)\b.{0,25}\b(user|source)\b", "evasion"),
    (r"\b(act|pretend|roleplay) as\b.{0,20}\b(admin|root|developer|dba)\b",
     "privilege_escalation"),
)

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in INJECTION_PATTERNS]


def scan_for_injection(text: str) -> list[dict]:
    """Return a list of ``{pattern, category, match}`` findings."""
    findings = []
    for regex, label in _COMPILED:
        for match in regex.finditer(text or ""):
            findings.append(
                {
                    "category": label,
                    "match": match.group(0)[:120],
                    "span": [match.start(), match.end()],
                }
            )
    return findings


def sanitize_document(text: str) -> tuple[str, list[dict]]:
    """
    Neutralise instruction-like content inside a retrieved document.

    We do not silently delete text (that would hide evidence during the demo);
    we replace the offending span with a visible marker, and we return the
    findings so the agent can log them and the notebook can show them.
    """
    findings = scan_for_injection(text)
    if not findings:
        return text, []

    cleaned = text
    for finding in sorted(findings, key=lambda f: -f["span"][0]):
        start, end = finding["span"]
        cleaned = (
            cleaned[:start]
            + f"[REDACTED: suspected {finding['category']} in source document]"
            + cleaned[end:]
        )
    return cleaned, findings


def check_user_input(question: str, max_length: int = 1000) -> dict:
    """
    Screen the *user's* question. We do not hard-block borderline phrasing —
    we flag it, log it, and let the system prompt do the refusing, except for
    clear secret-exfiltration attempts which we block outright.
    """
    question = question or ""
    if len(question) > max_length:
        return {
            "allowed": False,
            "reason": f"Question exceeds {max_length} characters.",
            "findings": [],
        }

    findings = scan_for_injection(question)
    blocking = [f for f in findings if f["category"] in
                {"secret_exfiltration", "privilege_escalation"}]
    if blocking:
        return {
            "allowed": False,
            "reason": "Request appears to target system secrets or privileged access.",
            "findings": findings,
        }
    return {"allowed": True, "reason": "ok", "findings": findings}
