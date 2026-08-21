"""Copy a date-bounded subset of a JSONL corpus into a new file.

Reads a corpus one line at a time, keeps the records whose `published_at`
year falls inside the requested window, and writes the ORIGINAL line through
unchanged. Nothing is re-serialised, so the output is a byte-identical
subset of the input.

Every rejected record is counted by reason and reconciled against the input
total, so nothing is dropped silently.

Run:
    python scripts/slice_by_date.py processed/articles_clean.jsonl \
        processed/recent_2021.jsonl --min-year 2021
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROGRESS_EVERY = 50_000


def record_year(record: dict) -> int | None:
    """Return the publication year, or None when it cannot be read.

    `published_at` looks like "2003-06-02T07:08:00+00:00", so the first four
    characters are the year. We deliberately do not parse the full timestamp:
    year granularity is all the window needs.
    """
    published = record.get("published_at") or ""
    year_text = published[:4]
    if len(year_text) != 4 or not year_text.isdigit():
        return None
    return int(year_text)


def slice_corpus(
    source: Path,
    destination: Path,
    min_year: int | None,
    max_year: int | None,
) -> tuple[Counter, Counter]:
    """Stream `source`, write matching lines to `destination`, count outcomes."""
    counts: Counter = Counter()
    years: Counter = Counter()

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

            year = record_year(record)
            if year is None:
                counts["undated"] += 1
                continue
            if min_year is not None and year < min_year:
                counts["too_old"] += 1
                continue
            if max_year is not None and year > max_year:
                counts["too_new"] += 1
                continue

            writer.write(raw_line)
            counts["written"] += 1
            years[year] += 1

    return counts, years


def report(counts: Counter, years: Counter, destination: Path) -> bool:
    """Print the outcome. Return True only if every input line is accounted for."""
    accounted = (counts["written"] + counts["too_old"] + counts["too_new"]
                 + counts["undated"] + counts["malformed"])

    print(f"\nsource lines read : {counts['read']:,}")
    print(f"  written         : {counts['written']:,}   -> {destination}")
    print(f"  before window   : {counts['too_old']:,}")
    print(f"  after window    : {counts['too_new']:,}")
    print(f"  undated         : {counts['undated']:,}")
    print(f"  malformed JSON  : {counts['malformed']:,}")
    print(f"  {'-' * 16}")
    print(f"  accounted for   : {accounted:,} of {counts['read']:,}")

    if years:
        print("\nwritten by year:")
        for year in sorted(years):
            print(f"  {year}  {years[year]:>7,}")

    return accounted == counts["read"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSONL corpus to read")
    parser.add_argument("destination", type=Path, help="JSONL file to write")
    parser.add_argument("--min-year", type=int, default=None,
                        help="Keep records published in this year or later")
    parser.add_argument("--max-year", type=int, default=None,
                        help="Keep records published in this year or earlier")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the destination if it already exists")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}")
    if args.min_year is None and args.max_year is None:
        raise SystemExit("Give at least one of --min-year / --max-year")
    if args.destination.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.destination} (use --force)")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    counts, years = slice_corpus(
        args.source, args.destination, args.min_year, args.max_year
    )
    if not report(counts, years, args.destination):
        raise SystemExit("COUNTS DO NOT RECONCILE -- do not use this output")


if __name__ == "__main__":
    main()


