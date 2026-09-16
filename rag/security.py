
import re


FORBIDDEN_SQL = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE"
]


def validate_sql(sql: str):

    normalized = sql.strip().upper()

    if not (
        normalized.startswith("SELECT")
        or normalized.startswith("WITH")
    ):
        return False, "Only SELECT/WITH queries are allowed."

    for keyword in FORBIDDEN_SQL:
        pattern = rf"\b{keyword}\b"

        if re.search(pattern, normalized):
            return False, f"Forbidden SQL operation: {keyword}"

    return True, "SQL is allowed."
