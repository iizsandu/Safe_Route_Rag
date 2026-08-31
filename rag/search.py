"""Rank articles for a query using BM25.

The online half of the system. Everything needed was computed offline by
rag.index; searching is arithmetic over the postings of the query's terms.

Only articles containing at least one query term are ever touched. Articles
sharing no word with the query are never loaded, never scored, never seen.

Run:
    python -m rag.search processed/dev_20k.jsonl asaram ashram deaths
    python -m rag.search processed/dev_20k.jsonl --explain      (interactive)
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rag.index import DocumentStore, InvertedIndex, build_index
from rag.tokenizer import tokenize

# Okapi BM25 defaults, unchanged since the original work and shipped by
# Lucene and Elasticsearch. Tuning them without an evaluation set would be
# fiddling rather than engineering; that belongs in Phase 8.
K1 = 1.5
B = 0.75

def inverse_document_frequency(doc_count: int, doc_containing: int) -> float:
    """How much evidence one match of this term provides.

    Lucene's variant. The +1 inside the log keeps the result positive even
    for a term appearing in more than half the corpus. The textbook form goes
    negative there, which would let a very common word SUBTRACT from a
    document's score -- an article punished for containing a word you asked
    for. With `said` in 80% of our articles, that is not hypothetical.
    """
    return math.log(1+ (doc_count - doc_containing + 0.5)/(doc_containing + 0.5))

@dataclass(frozen=True)
class TermStrength:
    """What one query term is worth, before any article is scored."""

    term: str
    doc_containing: int
    idf: float
    max_contribution: float
    in_index: bool

@dataclass(frozen=True)
class ScoreCeiling:
    """The highest score ANY article could reach for this query."""

    query: str
    terms: list[TermStrength]
    ceiling: float

    @property
    def unknown_terms(self) -> list[str]:
        """Terms absent from the index. They contribute exactly nothing."""
        return [term.term for term in self.terms if not term.in_index]

    @property
    def anchor(self) -> TermStrength | None:
        """The strongest single term -- what actually makes a query searchable.

        F-064: the SUM rewards word count, so "bus stop" (16.84) outranks
        "Rohini" (12.28) while locating nothing -- two mediocre words beat one
        good one. What discriminates is whether ONE term is rare enough to pin
        down a small set of articles.

        Reported ALONGSIDE `ceiling`, not instead of it. The sum is still the
        true upper bound on any article's score; it is simply the wrong thing
        to gate on.

        None when no query term is in the index at all.
        """
        known = [term for term in self.terms if term.in_index]
        return max(known, key=lambda term: term.idf) if known else None

def score_ceiling(index: InvertedIndex, query: str) -> ScoreCeiling:
    """The highest score any article could possibly reach for this query.

    Computable BEFORE searching: it needs only how many documents contain each
    query term, which the index already holds. No postings are scored and no
    document is read.

    Where 2.5 comes from. One term contributes

        idf * tf * (K1 + 1) / (tf + K1 * normalizer)

    As tf grows this approaches idf * (K1 + 1) = idf * 2.5 and never reaches
    it. Repetition saturates (F-043), so a term can never be worth more than
    2.5 x its idf however often an article repeats it. Document length changes
    how fast the limit is approached, never the limit itself -- which is why
    the ceiling needs no document lengths.

    A LOW ceiling means every article in the corpus is packed into a narrow
    band. The gaps that then decide the ranking are set by document length
    rather than relevance (F-047), so the top 5 are arbitrary while still
    looking like a ranking.

    This function REPORTS. It deliberately applies no threshold: none has been
    measured yet, and inventing one here would bury a guess inside a library.
    """
    # Same de-duplication as search(), for the same reason: "rohini rohini"
    # must not count twice. If the two disagreed, the ceiling would not bound
    # the thing it claims to bound.
    query_terms = list(dict.fromkeys(tokenize(query)))

    strengths: list[TermStrength] = []
    for term in query_terms:
        postings = index.postings.get(term)
        if postings is None:
            strengths.append(TermStrength(term, 0, 0.0, 0.0, in_index=False))
            continue
        idf = inverse_document_frequency(index.doc_count, len(postings))
        strengths.append(
            TermStrength(term, len(postings), idf, idf * (K1 + 1), in_index=True)
        )

    return ScoreCeiling(
        query=query,
        terms=strengths,
        ceiling=sum(strength.max_contribution for strength in strengths),
    )


@dataclass
class TermCOntribution:
    """What one query term contributed to one document's score.

    Exists purely for debuggability: it costs memory and buys nothing at
    runtime. CLAUDE.md section 7 requires a retrieval result to expose its
    own reasoning, so that a strange ranking can be explained rather than
    guessed at.
    """

    term: str
    term_frequency: int
    idf: float
    score: float

@dataclass
class SearchResult:
    doc_number: int
    score: float
    contributions: list[TermCOntribution] = field(default_factory=list) #to have contributions empty for new initializations

def search(
        index: InvertedIndex,
        query: str,
        limit: int
    ) -> tuple[list[SearchResult], list[str]]:
    """Return the top `limit` results, plus the tokens the query became."""
    # dict.fromkeys de-duplicates while preserving order. Without it, a query
    # of "police police" would score every article twice as high as "police"
    # -- same articles, same order, meaningless numbers.
    query_terms = list(dict.fromkeys(tokenize(query)))

    scores: dict[int, float] = defaultdict(float)
    contributions: dict[int, list[TermCOntribution]] = defaultdict(list)

    for term in query_terms:
        postings = index.postings.get(term)
        if postings is None:
            continue

        idf = inverse_document_frequency(index.doc_count, len(postings))

        for doc_number, term_frequency in postings:
            length_ratio = index.doc_lengths[doc_number] / index.avg_doc_length
            normalizer = 1 - B + B * length_ratio

            contribution = idf * (term_frequency * (K1 + 1)) / (
                term_frequency + K1 * normalizer
            )

            scores[doc_number] += contribution
            contributions[doc_number].append(
                TermCOntribution(term, term_frequency, idf, contribution)
            )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    results = [SearchResult(doc, score, contributions[doc]) for doc, score in ranked]
    return results, query_terms

def print_query_analysis(index: InvertedIndex, query_terms: list[str]) -> None:
    """Show what the query became, and how much each term is worth."""
    print(f"\nquery tokens : {query_terms}")

    for term in query_terms:
        postings = index.postings.get(term)
        if postings is None:
            print(f"  {term:<16} NOT IN INDEX -- contributes nothing")
            continue

        df = len(postings)
        idf = inverse_document_frequency(index.doc_count, df)
        share = 100 * df / index.doc_count
        print(f"  {term:<16} {df:>6,} articles ({share:5.2f}%)   idf={idf:5.2f}")


def print_results(
    results: list[SearchResult],
    store: DocumentStore,
    index: InvertedIndex,
    explain: bool,
) -> None:
    """Print the ranking, and optionally why each article scored what it did."""
    if not results:
        print("\nno matching articles")
        return

    for rank, result in enumerate(results, start=1):
        record = store.get(result.doc_number)
        length = index.doc_lengths[result.doc_number]

        print(f"\n{rank:>2}. {result.score:6.2f}  {record['headline']}")
        print(f"      {record['published_at'][:10]}  "
              f"{record['author'] or '(no author)'}  "
              f"id={record['article_id']}  {length} tokens")

        if record["snippet"]:
            print(f"      {record['snippet']}...")

        if explain:
            for item in sorted(result.contributions,
                               key=lambda c: c.score, reverse=True):
                print(f"        {item.term:<16} tf={item.term_frequency:<4} "
                      f"idf={item.idf:5.2f}  ->  {item.score:5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL slice to index")
    parser.add_argument("query", nargs="*",
                        help="Search terms. Omit for an interactive prompt.")
    parser.add_argument("--limit", type=int, default=10,
                        help="How many results to show")
    parser.add_argument("--snippet-chars", type=int, default=300,
                        help="Characters of body text kept per article")
    parser.add_argument("--explain", action="store_true",
                        help="Show each term's contribution to each score")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    # Article text contains non-ASCII; a cp1252 console would raise on it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"building index from {args.path} ...")
    index, store = build_index(args.path, args.snippet_chars)
    print(f"{index.doc_count:,} articles, {index.vocabulary_size:,} terms, "
          f"avgdl={index.avg_doc_length:.1f}")

    def run(query: str) -> None:
        results, query_terms = search(index, query, args.limit)
        print_query_analysis(index, query_terms)
        print_results(results, store, index, args.explain)

    if args.query:
        run(" ".join(args.query))
        return

    # Interactive: the index cost 13 seconds to build, so reuse it rather
    # than paying that again for every query.
    while True:
        try:
            query = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            return
        run(query)


if __name__ == "__main__":
    main()
