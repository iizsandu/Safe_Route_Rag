"""Measure the score ceiling for a list of place names.

Answers one question: where does "searchable" end and "too common to search"
begin? No threshold is applied here. The point is to SEE the spread first and
choose the line from it, rather than the other way round (CLAUDE.md section 10).

Run:
    python -m eval.measure_ceilings processed/recent_2021.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.index import build_index
from rag.search import ScoreCeiling, score_ceiling

# Deliberately mixed. The bad queries are not padding -- without them we would
# only learn that good queries look good, which is not a threshold.
DEFAULT_QUERIES = [
    # neighbourhoods -- expected searchable
    "Rohini", "Vijay Vihar", "Whitefield", "Koramangala", "Indiranagar",
    "Saket", "Dwarka", "Karol Bagh", "Marathahalli", "Electronic City",
    # cities -- expected too broad to locate anything
    "Delhi", "Bengaluru", "Mumbai", "Noida", "Gurgaon",
    # generic words a real user might type
    "road", "market", "station", "main road", "bus stop", "city centre",
    # rare token, ambiguous place: a known blind spot, not a success
    "MG Road",
    # does naming the city help, or dilute the rare word?
    "Rohini Delhi", "Whitefield Bengaluru",
    # should not exist in the corpus at all
    "Zzyzx",
]

def load_queries(path: Path | None) -> list[str]:
    """Queries from a file, one per line, or the built-in mix."""
    if path is None:
        return DEFAULT_QUERIES
    lines = path.read_text(encoding="utf-8").splitlines()
    stripped = (line.strip() for line in lines)
    # '#' comments let the query file record WHY each group is there -- which
    # names are expected to pass, which are deliberate failures, and which are
    # unverified. A list of bare strings loses all of that.
    return [line for line in stripped if line and not line.startswith("#")]

def format_terms(measured: ScoreCeiling) -> str:
    """Per-term detail, so a low ceiling can be attributed to a term."""
    if not measured.terms:
        return "(nothing survived tokenization)"
    parts = []
    for term in measured.terms:
        if term.in_index:
            parts.append(f"{term.term}({term.doc_containing:,} docs, idf {term.idf:.2f})")
        else:
            parts.append(f"{term.term}(ABSENT)")
    return "  ".join(parts)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL slice to index")
    parser.add_argument("--queries", type=Path, default=None,
                        help="File with one query per line. Defaults to a built-in mix.")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    # Article text is non-ASCII; a cp1252 console would raise on it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"building index from {args.path} ...")
    # snippet_chars=1: we never print an article here, so storing snippets
    # would cost memory for nothing.
    index, _ = build_index(args.path, snippet_chars=1)
    print(f"{index.doc_count:,} articles, {index.vocabulary_size:,} terms\n")

    # Sorted by ANCHOR, not by ceiling. F-064 measured the sum as unusable for
    # this decision: it rewards word count, so "bus stop" outranks "Rohini".
    measured = sorted(
        (score_ceiling(index, query) for query in load_queries(args.queries)),
        key=lambda result: result.anchor.idf if result.anchor else -1.0,
    )

    print(f"{'anchor':>7}  {'sum':>7}  {'query':<24}  terms")
    print("-" * 104)
    for result in measured:
        anchor = f"{result.anchor.idf:.2f}" if result.anchor else "--"
        print(f"{anchor:>7}  {result.ceiling:>7.2f}  {result.query:<24}  "
              f"{format_terms(result)}")

if __name__ == "__main__":
    main()

