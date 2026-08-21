"""Build an inverted index over a JSONL corpus slice.

Reads every article once, tokenizes it, and accumulates what BM25 needs:
which articles contain each token and how often, how long each article is,
and the average length.

Stores no article text. Fetching an article by id is a different job -- the
document store -- and arrives when search does.

Run:
    python -m rag.index processed/dev_20k.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from rag.tokenizer import tokenize

FULL_CORPUS_SIZE = 308_756
SNIPPET_CHARS = 300

@dataclass
class InvertedIndex:
    """Everything BM25 needs, and nothing else.

    postings[token] is a list of (doc_number, term_frequency) pairs, one per
    article containing that token. doc_number is a position into doc_lengths
    and article_ids, not an article_id.
    """

    postings: dict[str, list[tuple[int, int]]]
    doc_lengths: list[int]
    avg_doc_length: float

    @property
    def doc_count(self) -> int:
        return len(self.doc_lengths)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def posting_count(self) -> int:
        return sum(len(entries) for entries in self.postings.values())

@dataclass
class DocumentStore:
    """Display data for each article, addressed by internal doc number.

    Deliberately separate from InvertedIndex. This is the source-of-truth
    half; the index is derived and can always be rebuilt from it. Keeping
    them apart is what lets this move to disk later without the index
    changing at all.
    """

    article_ids: list[int]
    headlines: list[str]
    urls: list[str]
    published: list[str]
    authors: list[str]
    snippets: list[str]

    def get(self, doc_number: int) -> dict[str, Any]:
        return {
            "article_id": self.article_ids[doc_number],
            "headline": self.headlines[doc_number],
            "url": self.urls[doc_number],
            "published_at": self.published[doc_number],
            "author": self.authors[doc_number],
            "snippet": self.snippets[doc_number],
        }

def index_text(record: dict[str, Any]) -> str:
    """Return the text to index: headline, then body.

    `description` is deliberately excluded. F-024 found it is a truncated
    prefix of the body, so including it would count every word of the
    article's opening twice and distort term frequencies.
    """

    headline = record.get("headline") or ""
    body = record.get("body_text") or ""
    return f"{headline} {body}"

def build_index(
    path: Path,
    snippet_chars: int = SNIPPET_CHARS,
) -> tuple[InvertedIndex, DocumentStore]:
    """Stream the corpus once and build both the index and the store."""

    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lengths: list[int] = []
    article_ids: list[int] = []
    headlines: list[str] = []
    urls: list[str] = []
    published: list[str] = []
    authors: list[str] = []
    snippets: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for doc_number, raw_line in enumerate(handle):
            record = json.loads(raw_line)
            tokens = tokenize(index_text(record))

            for token, frequency in Counter(tokens).items():
                postings[token].append((doc_number, frequency))

            body = record.get("body_text") or ""

            doc_lengths.append(len(tokens))
            article_ids.append(record["article_id"])
            headlines.append(record.get("headline") or "")
            urls.append(record.get("url") or "")
            published.append(record.get("published_at") or "")
            authors.append(record.get("author") or "")
            # split() then join collapses newlines and runs of spaces, so the
            # snippet always prints as a single tidy line.
            snippets.append(" ".join(body[:snippet_chars].split()))

    average = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    index = InvertedIndex(dict(postings), doc_lengths, average)
    store = DocumentStore(
        article_ids, headlines, urls, published, authors, snippets
    )
    return index, store


def report(
    index: InvertedIndex,
    seconds: float,
    peak_bytes: int | None,
    top: int,
) -> None:
    """Print index statistics and a linear extrapolation to the full corpus.

    `peak_bytes` is None when the run was not traced. Memory lines are then
    omitted rather than printed as zero, so an untraced run can never be
    mistaken for a run that measured no memory.
    """
    scale = FULL_CORPUS_SIZE / index.doc_count if index.doc_count else 0.0

    print(f"documents      : {index.doc_count:,}")
    print(f"tokens total   : {sum(index.doc_lengths):,}")
    print(f"avg doc length : {index.avg_doc_length:.1f} tokens   <- BM25 avgdl")
    print(f"vocabulary     : {index.vocabulary_size:,} unique tokens")
    print(f"postings       : {index.posting_count:,}")
    print(f"build time     : {seconds:.1f} s")
    if peak_bytes is None:
        print("peak memory    : not measured   <- re-run with --trace")
    else:
        print(f"peak memory    : {peak_bytes / 1024 ** 2:,.0f} MB")

    print(f"\nextrapolated to {FULL_CORPUS_SIZE:,} articles (linear, pessimistic):")
    print(f"  postings     : {index.posting_count * scale / 1e6:,.0f} million")
    if peak_bytes is not None:
        print(f"  memory       : {peak_bytes * scale / 1024 ** 3:.1f} GB")
    print(f"  build time   : {seconds * scale / 60:.0f} min")

    print(f"\nwidest posting lists -- the tokens costing the most memory:")
    widest = sorted(index.postings.items(), key=lambda item: len(item[1]), reverse=True)
    for token, entries in widest[:top]:
        share = 100 * len(entries) / index.doc_count
        print(f"  {token:<14} {len(entries):>7,} articles  ({share:5.1f}%)")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL slice to index")
    parser.add_argument("--top", type=int, default=15,
                        help="How many widest posting lists to show")
    parser.add_argument("--trace", action="store_true",
                        help="Measure peak memory with tracemalloc. It traces "
                             "every allocation and costs several times the "
                             "build time, so it is off by default.")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.trace:
        tracemalloc.start()

    started = perf_counter()
    index, _store = build_index(args.path)
    elapsed = perf_counter() - started

    peak_bytes: int | None = None
    if args.trace:
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    report(index, elapsed, peak_bytes, args.top)


if __name__ == "__main__":
    main()