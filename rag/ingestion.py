"""
Document ingestion: load knowledge-base files and split them into chunks.

Design notes
------------
* Chunking is **markdown-header aware**. Our knowledge base is a set of small
  definition files where every `##` / `###` section is already a self-contained
  concept ("Average Order Value", "revenue column", ...). Splitting on headers
  gives far better retrieval than blind fixed-size windows, and it lets us cite
  a precise section instead of a whole file.
* Very long sections are additionally split with a sliding window so no chunk
  blows past the embedding model's context.
* Every chunk carries metadata (source file, section path, chunk id) because
  the project requires **RAG citations**.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

DEFAULT_KNOWLEDGE_FILES = (
    "business_definitions.md",
    "kpi_definitions.md",
    "data_dictionary.md",
    "analytics_guidelines.md",
)


@dataclass
class Chunk:
    """One retrievable unit of knowledge."""

    id: str
    content: str
    source: str            # file name, e.g. "kpi_definitions.md"
    section: str           # header path, e.g. "KPI Definitions > Average Order Value"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.section}" if self.section else self.source


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_documents(
    directory: str | Path = "data",
    filenames: Iterable[str] | None = DEFAULT_KNOWLEDGE_FILES,
    pattern: str = "*.md",
) -> list[dict]:
    """
    Read knowledge-base documents from disk.

    Parameters
    ----------
    directory : folder that holds the markdown knowledge files.
                In our repo the knowledge files live directly in `data/`.
    filenames : explicit allow-list. Pass ``None`` to glob every `*.md`
                in the folder instead (useful once the KB grows).
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(
            f"Knowledge directory '{root}' not found. "
            "Set KNOWLEDGE_DIR in .env or pass directory= explicitly."
        )

    if filenames is None:
        paths = sorted(root.glob(pattern))
    else:
        paths = [root / name for name in filenames if (root / name).exists()]

    if not paths:
        raise FileNotFoundError(f"No knowledge documents found in '{root}'.")

    documents = []
    for path in paths:
        documents.append(
            {
                "source": path.name,
                "path": str(path),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    return documents


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return a list of ``(section_path, body)`` tuples."""
    lines = text.splitlines()
    stack: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_path = ""
    buffer: list[str] = []

    for line in lines:
        match = _HEADER_RE.match(line)
        if match:
            if buffer and "".join(buffer).strip():
                sections.append((current_path, buffer))
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            current_path = " > ".join(stack)
            buffer = []
        else:
            buffer.append(line)

    if buffer and "".join(buffer).strip():
        sections.append((current_path, buffer))

    return [(path, "\n".join(body).strip()) for path, body in sections]


def sliding_window(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split oversized text on paragraph boundaries where possible."""
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start + int(chunk_size * 0.5), end)
            if cut == -1:
                cut = text.rfind(" ", start + int(chunk_size * 0.5), end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 900,
    overlap: int = 150,
    min_chars: int = 40,
) -> list[Chunk]:
    """Turn raw documents into a flat list of :class:`Chunk`."""
    chunks: list[Chunk] = []

    for doc in documents:
        source = doc["source"]
        for section_path, body in split_markdown_sections(doc["content"]):
            if len(body) < min_chars:
                continue
            for i, piece in enumerate(sliding_window(body, chunk_size, overlap)):
                # The section title is prepended so the embedding "knows" the
                # concept name even when the body never repeats it.
                content = f"{section_path}\n\n{piece}" if section_path else piece
                cid = hashlib.sha1(
                    f"{source}|{section_path}|{i}|{piece[:80]}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        id=cid,
                        content=content,
                        source=source,
                        section=section_path,
                        metadata={"part": i, "chars": len(piece)},
                    )
                )

    if not chunks:
        raise ValueError("Chunking produced 0 chunks — check the knowledge files.")
    return chunks


def build_corpus(directory: str | Path = "data", **kwargs) -> list[Chunk]:
    """Convenience: load + chunk in one call."""
    return chunk_documents(load_documents(directory), **kwargs)
