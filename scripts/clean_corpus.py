"""Produce a clean corpus from the raw article export.

Reads the raw JSONL and writes two files:
    processed/articles_clean.jsonl        records fit for indexing
    processed/articles_quarantine.jsonl   records rejected, each with a reason

Every input record appears in exactly one output file. The raw input is never
modified.

Three transformations, and nothing else:
    1. Replace the SQL-escape artifact '' with a single quote
    2. Quarantine records the producer marked content_quality != "ok"
    3. Flag (but keep) records containing fused tokens, which cannot be repaired

Usage:
    python scripts/clean_corpus.py Data/articles_20260820_093655.jsonl --limit 1000
    python scripts/clean_corpus.py Data/articles_20260820_093655.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Identifies which cleaner produced a record, so a stale output file is
# recognisable after the rules change.
CLEAN_VERSION = "c1"

# The SQL single-quote escape, left in prose by a database layer that was never
# unescaped on read. Reversal is deterministic: "Shetty''s" -> "Shetty's".
SQL_ESCAPE = "''"
SQL_ESCAPE_REPLACEMENT = "'"

# After repair, a surviving '' is usually fine: old wire copy uses `` and '' as
# opening and closing quote marks (``out of context''). A genuine miss has the
# possessive shape, with a word attached on BOTH sides: Shetty''s.
#
#   Shetty''s   letter before, lowercase after  -> possessive  -> report it
#    ''their    space before                    -> opening quote
#   context''   space after                     -> closing quote
UNFIXED_POSSESSIVE = re.compile(r"[A-Za-z]''[a-z]")

# Text nodes joined without a separator: "on Monday" -> "onMonday". The leading
# \b excludes CamelCase brand names, which start with a capital (WhatsApp).
# Cannot be repaired -- splitting "onMonday" reliably needs a dictionary and
# would corrupt proper nouns. We flag and keep.
FUSED_TOKEN = re.compile(r"\b[a-z]{2,}[A-Z][a-z]{2,}")

# Fields carrying prose that needs repair.
TEXT_FIELDS: tuple[str, ...] = ("headline", "description", "body_text")

MAX_EXAMPLES = 10
EXAMPLE_CONTEXT_CHARS = 30


@dataclass
class CleanStats:
    """Accumulates counts across a streamed pass over the corpus."""

    read: int = 0
    parse_failures: int = 0
    written_clean: int = 0
    quarantined: Counter[str] = field(default_factory=Counter)
    records_repaired: int = 0
    escapes_repaired: int = 0
    records_fused: int = 0
    unfixed_possessives: int = 0
    repair_examples: list[str] = field(default_factory=list)


def parse_line(raw_line: str) -> dict[str, Any] | None:
    """Return the decoded record, or None if the line is not a JSON object."""
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def iter_lines(path: Path, limit: int | None) -> Iterator[str]:
    """Yield non-blank lines, up to `limit`."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                return
            stripped = raw_line.strip()
            if stripped:
                yield stripped


def terminal_safe(text: str) -> str:
    """Escape non-printable characters so a terminal cannot execute them."""
    return "".join(ch if ch.isprintable() else f"\\x{ord(ch):02x}" for ch in text)


def repair_sql_escapes(text: str) -> tuple[str, int]:
    """Return the repaired text and how many escapes were replaced."""
    count = text.count(SQL_ESCAPE)
    if count == 0:
        return text, 0
    return text.replace(SQL_ESCAPE, SQL_ESCAPE_REPLACEMENT), count


def escape_contexts(text: str) -> Iterator[str]:
    """Yield the surrounding text for each SQL escape, for human review."""
    for match in re.finditer(re.escape(SQL_ESCAPE), text):
        start = max(0, match.start() - EXAMPLE_CONTEXT_CHARS)
        end = min(len(text), match.end() + EXAMPLE_CONTEXT_CHARS)
        yield text[start:end]


def quarantine_reason(record: dict[str, Any]) -> str | None:
    """Return why this record is unusable, or None if it should be kept."""
    quality = record.get("content_quality")
    if quality is None:
        return "missing_content_quality"
    if quality != "ok":
        return str(quality)
    if not (record.get("body_text") or "").strip():
        return "empty_body"  # producer said ok, but there is nothing to index
    return None


def clean_record(record: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return a repaired copy of the record and the number of repairs made."""
    cleaned = dict(record)
    repairs = 0

    for field_name in TEXT_FIELDS:
        value = cleaned.get(field_name)
        if isinstance(value, str):
            repaired_text, count = repair_sql_escapes(value)
            cleaned[field_name] = repaired_text
            repairs += count

    body = cleaned.get("body_text") or ""
    headline = cleaned.get("headline") or ""

    # Repair changes the length, so the precomputed field must be recomputed
    # or it silently disagrees with the text it describes.
    cleaned["body_chars"] = len(body)
    cleaned["has_fused_tokens"] = bool(FUSED_TOKEN.search(f"{headline}\n{body}"))
    cleaned["sql_escapes_repaired"] = repairs
    cleaned["clean_version"] = CLEAN_VERSION
    return cleaned, repairs


def collect_examples(stats: CleanStats, record: dict[str, Any]) -> None:
    """Capture a few pre-repair contexts so the assumption can be checked."""
    if len(stats.repair_examples) >= MAX_EXAMPLES:
        return
    for snippet in escape_contexts(record.get("body_text") or ""):
        stats.repair_examples.append(snippet)
        if len(stats.repair_examples) >= MAX_EXAMPLES:
            return


def clean_corpus(
    source: Path, clean_path: Path, quarantine_path: Path, limit: int | None
) -> CleanStats:
    """Stream the corpus once, writing clean and quarantined records."""
    stats = CleanStats()
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    with clean_path.open("w", encoding="utf-8") as clean_file, \
            quarantine_path.open("w", encoding="utf-8") as quarantine_file:

        for raw_line in iter_lines(source, limit):
            stats.read += 1
            record = parse_line(raw_line)
            if record is None:
                stats.parse_failures += 1
                continue

            reason = quarantine_reason(record)
            if reason is not None:
                stats.quarantined[reason] += 1
                record["quarantine_reason"] = reason
                quarantine_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            collect_examples(stats, record)
            cleaned, repairs = clean_record(record)

            if repairs:
                stats.records_repaired += 1
                stats.escapes_repaired += repairs
            if cleaned["has_fused_tokens"]:
                stats.records_fused += 1
            if UNFIXED_POSSESSIVE.search(cleaned.get("body_text") or ""):
                stats.unfixed_possessives += 1

            clean_file.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats.written_clean += 1

    return stats


def print_report(stats: CleanStats, clean_path: Path, quarantine_path: Path) -> None:
    """Print what the run did, in enough detail to verify it."""
    print(f"\n=== INPUT ===")
    print(f"records read        : {stats.read:,}")
    print(f"parse failures      : {stats.parse_failures:,}")

    total_quarantined = sum(stats.quarantined.values())
    print(f"\n=== OUTPUT ===")
    print(f"clean               : {stats.written_clean:,}  -> {clean_path}")
    print(f"quarantined         : {total_quarantined:,}  -> {quarantine_path}")
    accounted = stats.written_clean + total_quarantined + stats.parse_failures
    print(f"accounted for       : {accounted:,} of {stats.read:,} "
          f"{'OK' if accounted == stats.read else '*** MISMATCH ***'}")

    print(f"\n=== QUARANTINE REASONS ===")
    for reason, count in stats.quarantined.most_common():
        print(f"{count:>10,}  {terminal_safe(reason)}")

    print(f"\n=== REPAIRS ===")
    print(f"records repaired    : {stats.records_repaired:,}")
    print(f"escapes replaced    : {stats.escapes_repaired:,}")
    print(f"unfixed possessives : {stats.unfixed_possessives:,} "
          f"{'OK' if stats.unfixed_possessives == 0 else '*** CHECK THIS ***'}")

    print(f"\n=== FLAGGED, NOT REPAIRED ===")
    share = 100 * stats.records_fused / stats.written_clean if stats.written_clean else 0
    print(f"has_fused_tokens    : {stats.records_fused:,} ({share:.2f}%)")

    print(f"\n=== REPAIR EXAMPLES (before) — do these all look like possessives? ===")
    for snippet in stats.repair_examples:
        print(f"  ...{terminal_safe(snippet)}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the raw JSONL corpus")
    parser.add_argument("--clean-out", type=Path,
                        default=Path("processed/articles_clean.jsonl"))
    parser.add_argument("--quarantine-out", type=Path,
                        default=Path("processed/articles_quarantine.jsonl"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N lines (fast smoke test)")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stats = clean_corpus(args.source, args.clean_out, args.quarantine_out, args.limit)
    print_report(stats, args.clean_out, args.quarantine_out)


if __name__ == "__main__":
    main()
