"""Print one article verbatim for encoding forensics.

A terminal is itself a lossy encoder: characters the console cannot render are
dropped or substituted on the way to the screen. Text that looks broken on
screen may be intact in the file, and vice versa.

This tool prints each field twice -- rendered, and as repr(), which escapes
every non-ASCII character as \\uXXXX and is therefore console-safe.

Usage:
    python scripts/inspect_article.py Data/articles_20260818_100629.jsonl 105823902
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_BODY_CHARS = 500


def find_article(path: Path, article_id: int) -> dict[str, Any] | None:
    """Return the first record with the given article_id, or None if absent.

    A cheap substring test skips json.loads() on the vast majority of lines;
    the parsed record is then verified properly, so an over-matching prefilter
    cannot produce a wrong result.
    """
    needle = str(article_id)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if needle not in raw_line:
                continue
            record = json.loads(raw_line)
            if record.get("article_id") == article_id:
                return record
    return None


def print_field(name: str, value: Any) -> None:
    """Print one field rendered, then escaped, so the two can be compared."""
    print(f"\n--- {name} ---")
    print(f"rendered : {value}")
    print(f"repr     : {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the JSONL corpus")
    parser.add_argument("article_id", type=int, help="article_id to inspect")
    parser.add_argument("--body-chars", type=int, default=DEFAULT_BODY_CHARS,
                        help="How many characters of body_text to show")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    record = find_article(args.path, args.article_id)
    if record is None:
        raise SystemExit(f"article_id {args.article_id} not found")

    print(f"article_id : {record.get('article_id')}")
    print(f"url        : {record.get('url')}")
    print(f"published  : {record.get('published_at')}")

    print_field("headline", record.get("headline"))
    print_field("description", record.get("description"))
    print_field("body_text (truncated)",
                (record.get("body_text") or "")[:args.body_chars])


if __name__ == "__main__":
    main()
