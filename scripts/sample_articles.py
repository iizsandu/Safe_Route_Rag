"""Draw a uniform random sample of articles from the JSONL corpus.

Uses reservoir sampling so the file is read exactly once, in constant memory,
without needing to know the line count in advance.

Usage:
    python scripts/sample_articles.py Data/articles_20260818_100629.jsonl
    python scripts/sample_articles.py Data/... --size 20 --url-contains entertainment
    python scripts/sample_articles.py Data/... --seed 7 --body-preview 600
    python scripts/sample_articles.py processed/articles_clean.jsonl --size 500 \
        --out processed/sample_500.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_SAMPLE_SIZE = 30
DEFAULT_BODY_PREVIEW_CHARS = 400
DEFAULT_SEED = 42


def parse_line(raw_line: str) -> dict[str, Any] | None:
    """Return the decoded record, or None if the line is not a valid JSON object."""
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def matches_filter(record: dict[str, Any], url_substring: str | None) -> bool:
    """True if the record's URL contains the requested substring (or no filter)."""
    if url_substring is None:
        return True
    return url_substring.lower() in (record.get("url") or "").lower()


def url_section(url: str) -> str:
    """Return the leading path segments of a URL, e.g. 'city/delhi'."""
    segments = [seg for seg in urlparse(url).path.split("/") if seg]
    return "/".join(segments[:2]) if segments else "(empty)"


def reservoir_sample(
    path: Path,
    sample_size: int,
    url_substring: str | None,
    rng: random.Random,
) -> tuple[list[tuple[str, dict[str, Any]]], int, int]:
    """Stream the corpus and return (sample, matched_count, total_count).

    Each sample entry is a (raw_line, record) pair. The raw line is kept so the
    sample can be written back out byte-for-byte; re-serializing via json.dumps
    would alter escaping and misrepresent the very corruption we inspect for.

    Implements Algorithm R: the first `sample_size` matching records fill the
    reservoir; each subsequent record at position i (0-based) replaces a random
    reservoir slot with probability sample_size / (i + 1). Every matching record
    ends up equally likely to be in the final sample.
    """
    reservoir: list[tuple[str, dict[str, Any]]] = []
    matched_count = 0
    total_count = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_count += 1

            record = parse_line(stripped)
            if record is None or not matches_filter(record, url_substring):
                continue

            if len(reservoir) < sample_size:
                reservoir.append((stripped, record))
            else:
                slot = rng.randrange(matched_count + 1)
                if slot < sample_size:
                    reservoir[slot] = (stripped, record)
            matched_count += 1

    return reservoir, matched_count, total_count


def print_article(index: int, record: dict[str, Any], preview_chars: int) -> None:
    """Print one article in a form suitable for human review."""
    url = record.get("url") or ""
    body = record.get("body_text") or ""

    print(f"\n{'=' * 78}")
    print(f"[{index}] article_id={record.get('article_id')}  "
          f"section={url_section(url)}  published={record.get('published_at')}")
    print(f"HEADLINE : {record.get('headline')}")
    print(f"AUTHOR   : {record.get('author')}")
    print(f"URL      : {url}")
    print(f"BODY ({len(body)} chars):")
    print(body[:preview_chars] + ("..." if len(body) > preview_chars else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the JSONL corpus")
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Number of articles to sample")
    parser.add_argument("--url-contains", default=None,
                        help="Only sample articles whose URL contains this substring")
    parser.add_argument("--body-preview", type=int,
                        default=DEFAULT_BODY_PREVIEW_CHARS,
                        help="Characters of body text to display per article")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed, for reproducible samples")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write sampled records to this file as raw JSONL "
                             "instead of printing them")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")
    if args.size < 1:
        raise SystemExit("--size must be at least 1")

    # Article text contains non-ASCII and mojibake; a cp1252 console would raise.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample, matched, total = reservoir_sample(
        args.path, args.size, args.url_contains, random.Random(args.seed)
    )

    if args.out is None:
        for index, (_raw_line, record) in enumerate(sample, start=1):
            print_article(index, record, args.body_preview)
    else:
        # newline="\n" defeats Windows CRLF translation, so the written lines
        # stay byte-identical to the corpus they were drawn from.
        with args.out.open("w", encoding="utf-8", newline="\n") as out_handle:
            for raw_line, _record in sample:
                out_handle.write(raw_line + "\n")

    share = 100 * matched / total if total else 0
    print(f"\n{'=' * 78}")
    print(f"filter        : {args.url_contains or '(none)'}")
    print(f"matched       : {matched:,} of {total:,} articles ({share:.2f}%)")
    print(f"sampled       : {len(sample)}  (seed={args.seed})")
    if args.out is not None:
        print(f"wrote         : {len(sample)} records -> {args.out}")

if __name__=="__main__":
    main()