"""Describe the schema of a JSONL corpus file.

Streams the file once and reports, for every key: which types it holds, how
often it is present, and how often it is empty. Also checks whether every
record carries the same set of keys, and shows any record that does not.

JSONL does not guarantee a uniform schema -- that is a property of a
particular file, not of the format. This script is how you find out.

Read-only. Never modifies the file it inspects.

Usage:
    python scripts/describe_schema.py processed/dev_20k.jsonl
    python scripts/describe_schema.py processed/articles_clean.jsonl
    python scripts/describe_schema.py Data/articles_20260820_093655.jsonl --max-records 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_CHARS = 55


@dataclass
class FieldStats:
    """What we learned about one key across every record that carried it."""

    types: Counter[str] = field(default_factory=Counter)
    present: int = 0
    empty: int = 0
    sample: str = ""


@dataclass
class ScanResult:
    """Everything one pass over the file produced."""

    records: int = 0
    parse_failures: int = 0
    fields: dict[str, FieldStats] = field(default_factory=dict)
    signatures: Counter[tuple[str, ...]] = field(default_factory=Counter)
    first_line_of: dict[tuple[str, ...], int] = field(default_factory=dict)
    truncated: bool = False


def type_name(value: Any) -> str:
    """Return a readable type name. None is reported as 'null', not 'NoneType'."""
    return "null" if value is None else type(value).__name__


def is_empty(value: Any) -> bool:
    """True for null, empty string, empty list or empty dict.

    Deliberately NOT true for 0 or False: those are real values. Using plain
    falsiness here would report `body_chars: 0` and `has_encoding_loss: false`
    as missing data, which is a different and much more alarming claim.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, dict)):
        return len(value) == 0
    return False


def scan(path: Path, sample_chars: int, max_records: int | None) -> ScanResult:
    """Stream the file once, accumulating per-key and per-schema statistics."""
    result = ScanResult()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                result.parse_failures += 1
                continue

            if not isinstance(record, dict):
                result.parse_failures += 1
                continue

            result.records += 1

            # The schema signature is the sorted key set. Sorting matters:
            # two records with the same keys in a different order are the
            # same schema, and must not be counted as two variants.
            signature = tuple(sorted(record))
            result.signatures[signature] += 1
            result.first_line_of.setdefault(signature, line_number)

            for key, value in record.items():
                stats = result.fields.setdefault(key, FieldStats())
                stats.types[type_name(value)] += 1
                stats.present += 1
                if is_empty(value):
                    stats.empty += 1
                elif not stats.sample:
                    stats.sample = str(value)[:sample_chars].replace("\n", " ")

            if max_records is not None and result.records >= max_records:
                result.truncated = True
                break

    return result


def print_fields(result: ScanResult) -> None:
    """One row per key: types held, how often present, how often empty."""
    print(f"\n{'KEY':<22} {'TYPES':<20} {'PRESENT':>9} {'EMPTY':>8}  SAMPLE")
    print("-" * 100)

    for key, stats in result.fields.items():
        types = ", ".join(name for name, _ in stats.types.most_common())
        missing = result.records - stats.present
        present = f"{stats.present:,}" + ("*" if missing else "")
        print(f"{key:<22} {types:<20} {present:>9} {stats.empty:>8,}  {stats.sample}")

    if any(result.records - s.present for s in result.fields.values()):
        print("\n* this key is absent from some records -- see schema variants below")


def print_signatures(result: ScanResult) -> None:
    """Report whether every record shares the same key set, and how they differ."""
    variants = result.signatures.most_common()
    print(f"\nschema variants : {len(variants)}")

    if len(variants) == 1:
        signature, count = variants[0]
        print(f"  every one of {count:,} records has the same {len(signature)} keys")
        return

    dominant = set(variants[0][0])

    for rank, (signature, count) in enumerate(variants, start=1):
        line = result.first_line_of[signature]
        share = 100 * count / result.records if result.records else 0
        print(f"\n  variant {rank}: {count:,} records ({share:.2f}%), "
              f"first at line {line:,}")

        if rank == 1:
            print(f"    {len(signature)} keys -- treated as the norm")
            continue

        keys = set(signature)
        missing = sorted(dominant - keys)
        extra = sorted(keys - dominant)
        print(f"    missing: {', '.join(missing) or '(none)'}")
        print(f"    extra  : {', '.join(extra) or '(none)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the JSONL file")
    parser.add_argument("--sample-chars", type=int, default=DEFAULT_SAMPLE_CHARS,
                        help="Characters of the sample value to show per key")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Stop after this many records (for a quick look "
                             "at a very large file)")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    # Article text contains non-ASCII; a cp1252 console would raise on it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result = scan(args.path, args.sample_chars, args.max_records)

    print(f"file            : {args.path}")
    print(f"records         : {result.records:,}")
    print(f"parse failures  : {result.parse_failures:,}")
    print(f"distinct keys   : {len(result.fields)}")

    print_fields(result)
    print_signatures(result)

    if result.truncated:
        print(f"\nNOTE: stopped after {result.records:,} records (--max-records). "
              f"Every statement above describes only that prefix, which is NOT "
              f"a random sample of the file.")


if __name__ == "__main__":
    main()
