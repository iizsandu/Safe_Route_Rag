"""Copy every record whose text mentions a term into a new JSONL file.

Builds the candidate pool for an evaluation area: given "rohini", produce every
article the retriever could plausibly return for it.

Matches on `headline` + `body_text` only -- the same fields rag/index.py
indexes (F-024). Writes the ORIGINAL line through unchanged, so the output is
a byte-identical subset.

Run:
    python scripts/filter_by_term.py processed/recent_2021.jsonl \
        processed/rohini_2021.jsonl --term rohini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROGRESS_EVERY = 25_000


def build_matcher(term: str) -> re.Pattern:
    """Compile a whole-word, case-insensitive matcher for `term`.

    Word boundaries stop `rohini` matching inside a longer word. `re.escape`
    keeps a term containing punctuation from being read as regex syntax.
    """
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def filter_corpus(
    source: Path,
    destination: Path,
    matcher: re.Pattern,
) -> Counter:
    """Stream `source`, write matching lines to `destination`, count outcomes."""
    counts: Counter = Counter()

    with source.open("r", encoding="utf-8") as reader, \
            destination.open("w", encoding="utf-8") as writer:
        for raw_line in reader:
            counts["read"] += 1
            if counts["read"] % PROGRESS_EVERY == 0:
                print(f"  ...{counts['read']:,} lines", file=sys.stderr)

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                counts["malformed"] += 1
                continue

            in_headline = bool(matcher.search(record.get("headline") or ""))
            in_body = bool(matcher.search(record.get("body_text") or ""))

            if not (in_headline or in_body):
                counts["no_match"] += 1
                continue

            writer.write(raw_line)
            counts["written"] += 1
            if in_headline and in_body:
                counts["both"] += 1
            elif in_headline:
                counts["headline_only"] += 1
            else:
                counts["body_only"] += 1

    return counts


def report(counts: Counter, term: str, destination: Path) -> bool:
    """Print the outcome. Return True only if every input line is accounted for."""
    accounted = counts["written"] + counts["no_match"] + counts["malformed"]

    print(f"\nterm              : '{term}'  (whole word, case-insensitive)")
    print(f"source lines read : {counts['read']:,}")
    print(f"  written         : {counts['written']:,}   -> {destination}")
    print(f"  no match        : {counts['no_match']:,}")
    print(f"  malformed JSON  : {counts['malformed']:,}")
    print(f"  {'-' * 16}")
    print(f"  accounted for   : {accounted:,} of {counts['read']:,}")

    print("\nwhere the term appears:")
    print(f"  headline + body : {counts['both']:,}")
    print(f"  headline only   : {counts['headline_only']:,}")
    print(f"  body only       : {counts['body_only']:,}")

    return accounted == counts["read"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSONL corpus to read")
    parser.add_argument("destination", type=Path, help="JSONL file to write")
    parser.add_argument("--term", required=True,
                        help="Term to match, whole-word and case-insensitive")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the destination if it already exists")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}")
    if args.destination.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.destination} (use --force)")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    counts = filter_corpus(args.source, args.destination, build_matcher(args.term))
    if not report(counts, args.term, args.destination):
        raise SystemExit("COUNTS DO NOT RECONCILE -- do not use this output")


if __name__ == "__main__":
    main()
