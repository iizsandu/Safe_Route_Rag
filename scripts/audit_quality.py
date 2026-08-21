"""Audit a corpus against the producer's own quality claims.

Written for articles_20260820_093655.jsonl, which adds these fields:
    cities, keywords, content_quality, has_encoding_loss, parser_version

Also runs against the older 8-field schema; absent fields report as "(absent)".

The central question: how many records the producer marks content_quality="ok"
fail an independent check? Self-reported quality is a hypothesis, not a
measurement.

Usage:
    python scripts/audit_quality.py Data/articles_20260820_093655.jsonl --limit 1000
    python scripts/audit_quality.py Data/articles_20260820_093655.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# --- Defect signatures -------------------------------------------------------

# Standard SQL single-quote escape. Its presence in prose means text passed
# through a SQL layer without being unescaped on read.
SQL_ESCAPE = "''"

# UI controls and promo trailers that belong to the page, not the article.
UI_MARKERS: tuple[str, ...] = (
    "Read More",
    "You Can Also Check:",
    "Get real-time updates",
)

# Text nodes joined without a separator turn "on Monday" into "onMonday".
#
# STRICT requires the fused word to START with a lowercase letter (\b). That is
# what separates a real defect from a deliberately CamelCased brand name:
#     onMonday   -> starts lowercase -> counted
#     WhatsApp   -> starts uppercase -> ignored
# It under-reports, missing fusions whose first word was capitalised
# ("ChiefMinister"). Under-reporting is the safe direction for a figure we
# intend to hand back to the data producer.
FUSED_STRICT = re.compile(r"\b[a-z]{2,}[A-Z][a-z]{2,}")

# The naive pattern: any internal capital. Retained ONLY so the report can show
# the gap between the two counts, which is the brand-name false-positive rate.
FUSED_LOOSE = re.compile(r"[a-z]{2,}[A-Z][a-z]{2,}")

WORD = re.compile(r"[a-z]+")

SHORT_BODY_CHARS = 300      # below this, a "news article" is suspicious
MIN_HEADLINE_WORD = 4       # ignore short function words when testing overlap
MAX_EXAMPLES = 5            # article_ids to record per defect, for follow-up


@dataclass
class QualityAudit:
    """Accumulates audit statistics across a streamed pass over the corpus."""

    total: int = 0
    parse_failures: int = 0

    # The producer's own claims
    content_quality: Counter[str] = field(default_factory=Counter)
    encoding_loss: Counter[str] = field(default_factory=Counter)
    parser_version: Counter[str] = field(default_factory=Counter)

    # The new metadata fields
    city_counts: Counter[str] = field(default_factory=Counter)
    cities: Counter[str] = field(default_factory=Counter)
    keyword_counts: Counter[str] = field(default_factory=Counter)
    keywords: Counter[str] = field(default_factory=Counter)

    # Independent checks, counted ONLY on records the producer calls "ok"
    ok_total: int = 0
    ok_failing: Counter[str] = field(default_factory=Counter)
    ok_any_defect: int = 0
    ok_fused_loose: int = 0  # naive pattern, for false-positive comparison only

    # Corpus-wide defect rates, for cross-tabulation
    year_totals: Counter[int] = field(default_factory=Counter)
    sql_escape_by_year: Counter[int] = field(default_factory=Counter)
    fused_by_year: Counter[int] = field(default_factory=Counter)

    # Evidence
    fused_examples: Counter[str] = field(default_factory=Counter)
    example_ids: dict[str, list[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )


def parse_line(raw_line: str) -> dict[str, Any] | None:
    """Return the decoded record, or None if the line is not a JSON object."""
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def iter_lines(path: Path, limit: int | None) -> Iterator[str]:
    """Yield non-blank lines, up to `limit`."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                return
            stripped = raw_line.strip()
            if stripped:
                yield stripped


def terminal_safe(text: str) -> str:
    """Escape non-printable characters so a terminal cannot execute them.

    Learned the hard way: C1 control characters in corpus text are interpreted
    by terminals and silently delete surrounding output, including our own.
    """
    return "".join(ch if ch.isprintable() else f"\\x{ord(ch):02x}" for ch in text)


def parse_year(published_at: Any) -> int | None:
    """Return the publication year, or None if missing or unparseable."""
    if not isinstance(published_at, str):
        return None
    try:
        return datetime.fromisoformat(published_at).year
    except ValueError:
        return None


def headline_overlaps_body(headline: str, body: str) -> bool:
    """True if any substantial headline word also appears in the body.

    False is the strong signal that body_text belongs to a different article
    (DS-005). Substring matching is deliberate: it over-reports overlap, so
    the check errs towards NOT flagging, which keeps false alarms low.
    """
    words = [w for w in WORD.findall(headline.lower()) if len(w) >= MIN_HEADLINE_WORD]
    if not words:
        return True  # nothing testable; do not flag
    body_lower = body.lower()
    return any(word in body_lower for word in words)


def run_checks(record: dict[str, Any], body: str, headline: str) -> list[str]:
    """Return the names of every independent check this record fails."""
    failures: list[str] = []
    combined = f"{headline}\n{body}"

    if any(marker in combined for marker in UI_MARKERS):
        failures.append("ui_marker")
    if SQL_ESCAPE in combined:
        failures.append("sql_escape")
    if FUSED_STRICT.search(combined):
        failures.append("fused_tokens")
    if len(body) < SHORT_BODY_CHARS:
        failures.append("short_body")
    if not headline_overlaps_body(headline, body):
        failures.append("no_headline_overlap")
    return failures


def record_example(audit: QualityAudit, check: str, article_id: Any) -> None:
    """Keep up to MAX_EXAMPLES article_ids per check, for later inspection."""
    bucket = audit.example_ids[check]
    if len(bucket) < MAX_EXAMPLES:
        bucket.append(article_id)


def update_metadata_stats(audit: QualityAudit, record: dict[str, Any]) -> None:
    """Profile the producer's flags and the new metadata fields."""
    audit.content_quality[record.get("content_quality", "(absent)")] += 1
    audit.encoding_loss[str(record.get("has_encoding_loss", "(absent)"))] += 1
    audit.parser_version[record.get("parser_version", "(absent)")] += 1

    cities = record.get("cities")
    if isinstance(cities, list):
        audit.city_counts[str(len(cities)) if len(cities) < 3 else "3+"] += 1
        for city in cities:
            audit.cities[str(city)] += 1
    else:
        audit.city_counts["(absent)"] += 1

    keywords = record.get("keywords")
    if isinstance(keywords, list):
        audit.keyword_counts[str(len(keywords)) if len(keywords) < 3 else "3+"] += 1
        for keyword in keywords:
            audit.keywords[str(keyword)] += 1
    else:
        audit.keyword_counts["(absent)"] += 1


def audit_corpus(path: Path, limit: int | None = None) -> QualityAudit:
    """Stream the corpus once and return the audit."""
    audit = QualityAudit()

    for raw_line in iter_lines(path, limit):
        audit.total += 1
        record = parse_line(raw_line)
        if record is None:
            audit.parse_failures += 1
            continue

        update_metadata_stats(audit, record)

        body = record.get("body_text") or ""
        headline = record.get("headline") or ""
        article_id = record.get("article_id")
        combined = f"{headline}\n{body}"

        # Corpus-wide rates by year
        year = parse_year(record.get("published_at"))
        if year is not None:
            audit.year_totals[year] += 1
            if SQL_ESCAPE in combined:
                audit.sql_escape_by_year[year] += 1
            if FUSED_STRICT.search(combined):
                audit.fused_by_year[year] += 1

        # Evidence for the fused-token check
        for token in FUSED_STRICT.findall(combined)[:3]:
            audit.fused_examples[token] += 1

        # The central question: do "ok" records survive independent checks?
        if record.get("content_quality") != "ok":
            continue

        audit.ok_total += 1
        if FUSED_LOOSE.search(combined):
            audit.ok_fused_loose += 1
        failures = run_checks(record, body, headline)
        for check in failures:
            audit.ok_failing[check] += 1
            record_example(audit, check, article_id)
        if failures:
            audit.ok_any_defect += 1

    return audit


def print_counter(title: str, counter: Counter, top: int, total: int) -> None:
    """Print a counter as count / percentage / label, most common first."""
    print(f"\n=== {title} ===")
    for label, count in counter.most_common(top):
        share = 100 * count / total if total else 0
        print(f"{count:>10,}  {share:>6.2f}%  {terminal_safe(str(label))}")
    if len(counter) > top:
        print(f"... and {len(counter) - top:,} more distinct values")
    print(f"distinct values : {len(counter):,}")


def print_report(audit: QualityAudit) -> None:
    """Print the audit as a plain-text report."""
    total = audit.total

    print(f"\n=== VOLUME ===")
    print(f"records read   : {total:,}")
    print(f"parse failures : {audit.parse_failures:,}")

    print_counter("PRODUCER FLAG: content_quality", audit.content_quality, 15, total)
    print_counter("PRODUCER FLAG: has_encoding_loss", audit.encoding_loss, 5, total)
    print_counter("PRODUCER FLAG: parser_version", audit.parser_version, 10, total)

    print_counter("CITIES PER ARTICLE", audit.city_counts, 6, total)
    print_counter("TOP CITIES", audit.cities, 25, total)
    print_counter("KEYWORDS PER ARTICLE", audit.keyword_counts, 6, total)
    print_counter("TOP KEYWORDS", audit.keywords, 30, total)

    # The section that answers the question we actually care about.
    ok = audit.ok_total
    print(f"\n=== INDEPENDENT AUDIT OF content_quality == 'ok' ===")
    print(f"records marked 'ok' : {ok:,}")
    print(f"{'check':<24}{'failing':>12}{'share':>10}")
    for check in ("ui_marker", "sql_escape", "fused_tokens",
                  "short_body", "no_headline_overlap"):
        count = audit.ok_failing.get(check, 0)
        share = 100 * count / ok if ok else 0
        print(f"{check:<24}{count:>12,}{share:>9.2f}%")
    share_any = 100 * audit.ok_any_defect / ok if ok else 0
    print(f"{'ANY of the above':<24}{audit.ok_any_defect:>12,}{share_any:>9.2f}%")

    loose = audit.ok_fused_loose
    strict = audit.ok_failing.get("fused_tokens", 0)
    loose_share = 100 * loose / ok if ok else 0
    print(f"\nfused_tokens cross-check (the pattern that had the brand-name bug)")
    print(f"{'  naive pattern':<24}{loose:>12,}{loose_share:>9.2f}%")
    print(f"{'  strict pattern':<24}{strict:>12,}"
          f"{100 * strict / ok if ok else 0:>9.2f}%")
    print(f"{'  brand false positives':<24}{loose - strict:>12,}")

    print(f"\n=== EXAMPLE article_ids TO INSPECT ===")
    for check, ids in sorted(audit.example_ids.items()):
        print(f"{check:<24}{ids}")

    print(f"\n=== TOP FUSED TOKENS (evidence for DS-006) ===")
    for token, count in audit.fused_examples.most_common(30):
        print(f"{count:>8,}  {terminal_safe(token)}")
    print(f"distinct fused tokens : {len(audit.fused_examples):,}")

    print(f"\n=== DEFECT RATE BY YEAR ===")
    print(f"{'year':<6}{'articles':>10}{'sql_esc':>10}{'':>8}{'fused':>10}{'':>8}")
    for year in sorted(audit.year_totals):
        count = audit.year_totals[year]
        sql = audit.sql_escape_by_year.get(year, 0)
        fused = audit.fused_by_year.get(year, 0)
        print(f"{year:<6}{count:>10,}{sql:>10,}{100 * sql / count:>7.1f}%"
              f"{fused:>10,}{100 * fused / count:>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the JSONL corpus")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N lines (fast smoke test)")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print_report(audit_corpus(args.path, args.limit))


if __name__ == "__main__":
    main()
