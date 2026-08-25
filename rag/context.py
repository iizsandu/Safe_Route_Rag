"""Turn retrieved articles into the text an LLM reads.

Implements the source-block format fixed by D-017. Every element of that format
exists because something specific breaks without it -- read the ADR before
simplifying anything here.

Returns the article IDs alongside the text: D-018's fabrication check needs the
exact set of IDs that were sent, and re-deriving it elsewhere lets the two drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_OPEN = "=== SOURCE {number} ==="
SOURCE_CLOSE = "=== END SOURCE {number} ==="

# A body containing our own delimiter could close a source block early and have
# the rest of its text read as instructions. Retrieved text is untrusted data
# (CLAUDE.md section 9), so the marker is neutralised wherever it appears.
FORBIDDEN_IN_TEXT = "=== SOURCE", "=== END SOURCE"

EMPTY_BODY_MARKER = "(no body text in the corpus for this article)"

# A safety valve, not a truncation point. D-017 says full bodies; if a caller
# assembles something enormous we want a loud failure, not a silent trim.
MAX_CONTEXT_CHARS = 400_000

@dataclass
class Context:

    text: str
    article_ids: list[int]

    @property
    def source_count(self) -> int:
        return len(self.article_ids)

    @property
    def char_count(self) -> int:
        return len(self.text)

def neutralise_markers(text: str) -> str:
    """Defang any source delimiter appearing inside article text."""
    for marker in FORBIDDEN_IN_TEXT:
        text = text.replace(marker, marker.replace("=", "-"))
    return text

def format_source(number: int, record: dict[str, Any]) -> str:
    """Render one article as a delimited source block (D-017)."""
    if "article_id" not in record:
        raise ValueError(
            "record has no article_id; a source that cannot be cited must not "
            "enter a context (D-017)"
        )

    body = (record.get("body_text") or "").strip()
    text = neutralise_markers(body) if body else EMPTY_BODY_MARKER

    return "\n".join([
        SOURCE_OPEN.format(number=number),
        f"cite_as: S{number}",
        f"article_id: {record['article_id']}",
        f"published: {(record.get('published_at') or '')[:10]}",
        f"headline: {neutralise_markers(record.get('headline') or '')}",
        f"text: {text}",
        SOURCE_CLOSE.format(number=number),
    ])


def build_context(records: list[dict[str, Any]]) -> Context:
    """Assemble retrieved articles into one context.

    Sends exactly four fields per article. `cities` and `keywords` are excluded
    deliberately -- they are the search terms used to COLLECT the article, not a
    description of it, and a model reading them invents claims it can then cite
    with a real article id (F-053).
    """
    blocks = [format_source(number, record)
              for number, record in enumerate(records, start=1)]
    text = "\n\n".join(blocks)

    if len(text) > MAX_CONTEXT_CHARS:
        raise ValueError(
            f"context is {len(text):,} chars, over the {MAX_CONTEXT_CHARS:,} "
            f"guard. Send fewer sources rather than truncating (D-017)."
        )

    return Context(text=text, article_ids=[r["article_id"] for r in records])
