"""Profile the raw crime-news JSONL corpus.

Streams the file line by line so memory use stays flat regardless of corpus
size. Produces a factual report used to drive ingestion design decisions.

Usage:
    python scripts/profile_dataset.py Data/articles_20260818_100629.jsonl --limit 1000
    python scripts/profile_dataset.py Data/articles_20260818_100629.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

# Unicode replacement character. Its presence after repair means the original
# bytes were destroyed upstream and cannot be recovered.
REPLACEMENT_CHAR = "\ufffd"


# Upper bounds for body-length buckets, in characters.
LENGTH_BUCKETS: tuple[int, ...] = (0, 200, 500, 1000, 2000, 4000, 8000, 16000)


@dataclass
class FieldStats:
    """Completeness and type profile for a single JSON field."""

    present: int = 0
    null: int = 0
    empty: int = 0
    observed_types: Counter[str] = field(default_factory=Counter)


@dataclass
class CorpusProfile:
    """Accumulates statistics across a streamed pass over the corpus."""

    total_lines: int = 0
    parse_failures: list[int] = field(default_factory=list)
    key_signatures: Counter[frozenset[str]] = field(default_factory=Counter)
    fields: dict[str, FieldStats] = field(
        default_factory=lambda: defaultdict(FieldStats)
    )
    article_id_counts: Counter[Any] = field(default_factory=Counter)
    url_hosts: Counter[str] = field(default_factory=Counter)
    url_sections: Counter[str] = field(default_factory=Counter)
    authors: Counter[str] = field(default_factory=Counter)
    year_counts: Counter[int] = field(default_factory=Counter)
    unparseable_dates: int = 0
    length_buckets: Counter[str] = field(default_factory=Counter)
    body_chars_mismatches: int = 0
    double_encoded_articles: int = 0
    unrecoverable_articles: int = 0
    double_encoded_by_year: Counter[int] = field(default_factory=Counter)
    unrecoverable_by_year: Counter[int] = field(default_factory=Counter)


def parse_line(raw_line: str) -> dict[str, Any] | None:
    """Return the decoded record, or None if the line is not valid JSON."""
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def iter_lines(path: Path, limit: int | None) -> Iterator[tuple[int, str]]:
    """Yield (line_number, raw_line) for non-blank lines, up to `limit`."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                return
            stripped = raw_line.strip()
            if stripped:
                yield line_number, stripped


def bucket_label(length: int) -> str:
    """Map a character count onto a human-readable histogram bucket."""
    for lower, upper in zip(LENGTH_BUCKETS, LENGTH_BUCKETS[1:]):
        if lower <= length < upper:
            return f"{lower}-{upper}"
    return f"{LENGTH_BUCKETS[-1]}+"


def url_section(url: str) -> str:
    """Return the leading path segments of a URL, e.g. 'city/delhi'."""
    segments = [seg for seg in urlparse(url).path.split("/") if seg]
    return "/".join(segments[:2]) if segments else "(empty)"


def repair_mojibake(text: str) -> str:
    """Reverse UTF-8 bytes that were mistakenly decoded as latin-1.

    Returns the repaired text, or the input unchanged if it is not
    double-encoded. The repair is lossless when it applies.
    """
    try:
        candidate = text.encode("latin-1")
    except UnicodeEncodeError:
        return text  # contains characters outside latin-1: not double-encoded
    try:
        return candidate.decode("utf-8")
    except UnicodeDecodeError:
        return text  # bytes are not valid UTF-8: leave the text alone


def parse_year(published_at: Any) -> int | None:
    """Return the publication year, or None if missing or unparseable."""
    if not isinstance(published_at, str):
        return None
    try:
        return datetime.fromisoformat(published_at).year
    except ValueError:
        return None


def update_field_stats(profile: CorpusProfile, record: dict[str, Any]) -> None:
    """Record presence, nullity, emptiness and type for each field."""
    for key, value in record.items():
        stats = profile.fields[key]
        stats.present += 1
        stats.observed_types[type(value).__name__] += 1
        if value is None:
            stats.null += 1
        elif isinstance(value, (str, list, dict)) and len(value) == 0:
            stats.empty += 1


def update_derived_stats(profile: CorpusProfile, record: dict[str, Any]) -> None:
    """Record statistics that require interpreting specific known fields."""
    profile.article_id_counts[record.get("article_id")] += 1

    url = record.get("url") or ""
    if url:
        profile.url_hosts[urlparse(url).netloc] += 1
        profile.url_sections[url_section(url)] += 1

    profile.authors[record.get("author") or "(missing)"] += 1

    year = parse_year(record.get("published_at"))
    if year is None:
        profile.unparseable_dates += 1
    else:
        profile.year_counts[year] += 1

    body_text = record.get("body_text") or ""
    profile.length_buckets[bucket_label(len(body_text))] += 1
    if record.get("body_chars") != len(body_text):
        profile.body_chars_mismatches += 1

    headline = record.get("headline") or ""
    original_text = f"{headline}\n{body_text}"
    repaired_text = repair_mojibake(original_text)
    if repaired_text != original_text:
        profile.double_encoded_articles += 1
        if year is not None:
            profile.double_encoded_by_year[year] += 1
    if REPLACEMENT_CHAR in repaired_text:
        profile.unrecoverable_articles += 1
        if year is not None:
            profile.unrecoverable_by_year[year] += 1


def profile_corpus(path: Path, limit: int | None = None) -> CorpusProfile:
    """Stream the corpus once and return an aggregated profile."""
    profile = CorpusProfile()
    for line_number, raw_line in iter_lines(path, limit):
        profile.total_lines += 1
        record = parse_line(raw_line)
        if record is None:
            profile.parse_failures.append(line_number)
            continue
        profile.key_signatures[frozenset(record.keys())] += 1
        update_field_stats(profile, record)
        update_derived_stats(profile, record)
    return profile


def print_report(profile: CorpusProfile) -> None:
    """Print the profile as a plain-text report."""
    total = profile.total_lines
    parsed = total - len(profile.parse_failures)

    print(f"\n=== VOLUME ===")
    print(f"lines read        : {total:,}")
    print(f"parsed ok         : {parsed:,}")
    print(f"parse failures    : {len(profile.parse_failures):,} "
          f"{profile.parse_failures[:5]}")

    print(f"\n=== SCHEMA SIGNATURES (distinct key sets) ===")
    for keys, count in profile.key_signatures.most_common(5):
        print(f"{count:>9,}  {sorted(keys)}")

    print(f"\n=== FIELD COMPLETENESS ===")
    print(f"{'field':<16}{'present':>10}{'null':>10}{'empty':>10}  types")
    for name, stats in sorted(profile.fields.items()):
        types = ",".join(sorted(stats.observed_types))
        print(f"{name:<16}{stats.present:>10,}{stats.null:>10,}"
              f"{stats.empty:>10,}  {types}")

    duplicates = {i: c for i, c in profile.article_id_counts.items() if c > 1}
    print(f"\n=== IDENTITY ===")
    print(f"distinct article_id : {len(profile.article_id_counts):,}")
    print(f"duplicated ids      : {len(duplicates):,}")

    print(f"\n=== SOURCES ===")
    for host, count in profile.url_hosts.most_common(10):
        print(f"{count:>9,}  {host}")

    print(f"\n=== URL SECTIONS (top 20) ===")
    for section, count in profile.url_sections.most_common(20):
        print(f"{count:>9,}  {section}")
    print(f"distinct sections   : {len(profile.url_sections):,}")

    print(f"\n=== AUTHORS (top 10) ===")
    for author, count in profile.authors.most_common(10):
        print(f"{count:>9,}  {author}")
    print(f"distinct authors    : {len(profile.authors):,}")

    print(f"\n=== TIME COVERAGE AND DAMAGE BY YEAR ===")
    print(f"{'year':<6}{'articles':>10}{'dbl-enc':>10}{'':>8}{'lost':>9}{'':>8}")
    for year in sorted(profile.year_counts):
        count = profile.year_counts[year]
        encoded = profile.double_encoded_by_year.get(year, 0)
        lost = profile.unrecoverable_by_year.get(year, 0)
        print(f"{year:<6}{count:>10,}{encoded:>10,}{100 * encoded / count:>7.1f}%"
              f"{lost:>9,}{100 * lost / count:>7.1f}%")
    print(f"unparseable dates   : {profile.unparseable_dates:,}")

    print(f"\n=== BODY LENGTH (chars) ===")
    for label in [f"{a}-{b}" for a, b in
                  zip(LENGTH_BUCKETS, LENGTH_BUCKETS[1:])] + \
                 [f"{LENGTH_BUCKETS[-1]}+"]:
        print(f"{label:<12}{profile.length_buckets.get(label, 0):>9,}")

    print(f"\n=== DATA QUALITY FLAGS ===")
    print(f"body_chars != len(body_text) : {profile.body_chars_mismatches:,}")
    double_pct = 100 * profile.double_encoded_articles / parsed if parsed else 0
    lost_pct = 100 * profile.unrecoverable_articles / parsed if parsed else 0
    print(f"double-encoded (repairable)  : "
          f"{profile.double_encoded_articles:,} ({double_pct:.1f}%)")
    print(f"contains U+FFFD (bytes lost) : "
          f"{profile.unrecoverable_articles:,} ({lost_pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the JSONL corpus")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only read the first N lines (use for a fast smoke test)",
    )
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    print_report(profile_corpus(args.path, args.limit))


if __name__ == "__main__":
    main()
