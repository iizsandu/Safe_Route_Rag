"""Answer a question about an area, end to end.

Joins the two halves that already existed but were never connected: `rag.search`
finds candidate articles, `rag.generate` turns them into structured incidents.
Until now the article ids were typed by hand, which meant a person was doing the
retrieval and calling it a pipeline.

Two ways in, and keeping both is the point:

    rag.pipeline   --area "Rohini, Delhi"    searches all 107,264 articles.
                                             Real. NOT measurable -- nobody has
                                             labelled what it retrieves.
    rag.generate   --ids 1,2,3               the 20 hand-labelled articles.
                                             Measurable, and what every recall
                                             number in FINDINGS is computed
                                             against (F-057).

If searching simply replaced the fixed set, recall 0.68 would quietly start
describing a different set of articles and the answer key would stop applying
without anything appearing to break.

Run:
    python -m rag.pipeline processed/recent_2021.jsonl --area "Rohini, Delhi" --dry-run
    python -m rag.pipeline processed/recent_2021.jsonl --area "Rohini, Delhi"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag.embed import MODEL_NAME, embed_query, load_model
from rag.generate import generate, present
from rag.index import build_index
from rag.llm import ModelConfig
from rag.articles import connect as connect_articles
from rag.qdrant_search import connect as qdrant_connect
from rag.qdrant_search import date_bound
from rag.qdrant_search import search as qdrant_search
from rag.search import search
from rag.vectors import load_store
from rag.vectors import search as vector_search

def fetch_bodies(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    """Read the full record for specific article ids.

    Why this exists at all: `DocumentStore` keeps a 300-character snippet, and
    F-052 measured that truncating there loses the place name in 64% of
    articles. The store therefore cannot feed generation, which needs the full
    body (D-017).

    Two ways to fix that -- hold every body in memory to use ten of them, or
    stream the file again for the ten. This takes the second. It costs a
    second pass over the corpus and keeps memory flat, which is D-002's rule.

    The real answer is a disk-backed store addressed by byte offset, which
    `DocumentStore`'s own docstring already anticipates. That is a larger
    change than wiring the pipeline deserves.
    """
    found: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            article_id = record.get("article_id")
            if article_id in wanted:
                found[article_id] = record
                # Every id was found; the rest of the file cannot matter.
                if len(found) == len(wanted):
                    break
    return found

RRF_K = 60

def fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: combine rankings by POSITION, never by score.

    BM25 scores run 0-15 and cosine runs 0.4-0.9 (F-066, F-069). They measure
    different things on different scales, so adding or averaging them is
    meaningless -- and normalising them first only hides the assumption that
    the two scales are comparable, which nothing supports.

    RRF sidesteps it: each retriever contributes 1/(k + rank). Only order
    matters. k=60 is the value from the original paper, and it damps the top
    few positions so a single retriever's confident wrong answer cannot
    dominate -- which is exactly the failure F-071 found in BM25's top 10.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, article_id in enumerate(ranking, start=1):
            scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: -item[1])

def retrieve(args) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    """Rank articles. Returns (ranked, records).

    `records` is empty on the FAISS path and populated on the Qdrant path --
    Qdrant returns the article text alongside the vector, so the 292 MB corpus
    file never has to be re-read per request (D-031).
    """
    # Deeper than --limit so fusion has something to work with: an article
    # ranked 30th by one retriever and 2nd by the other should surface.
    depth = max(args.limit * 5, 50)
    rankings: list[list[int]] = []
    dense_scores: list[tuple[int, float]] = []
    records: dict[int, dict] = {}

    # The production path: one request, both retrievers, fused inside Qdrant,
    # article text returned with the results. No index to build, no file to
    # re-read (D-031).
    if args.store == "qdrant" and args.retriever == "hybrid":
        since = date_bound(args.searched_from)
        until = date_bound(args.searched_to)
        hits = qdrant_search(qdrant_connect(args.qdrant_url), connect_articles(),
                             embed_query(args.area), args.area, depth,
                             since=since, until=until, fuse=fuse)
        print(f"qdrant: {len(hits)} articles in {args.searched_from} to "
              f"{args.searched_to}, keyword + meaning, text included")
        return ([(article_id, score) for article_id, score, _ in hits],
                {article_id: payload for article_id, _, payload in hits})

    # Everything below is for COMPARISON, not for serving: --retriever bm25 or
    # dense in isolation, and --store faiss as the exact reference (F-077).
    if args.retriever in ("bm25", "hybrid"):
        print(f"building BM25 index from {args.path} ...")
        index, store = build_index(args.path, snippet_chars=1)
        print(f"  {index.doc_count:,} articles, {index.vocabulary_size:,} terms")
        results, query_terms = search(index, args.area, depth)
        print(f"  query tokens: {query_terms}")
        rankings.append([store.get(r.doc_number)["article_id"] for r in results])

    if args.retriever in ("dense", "hybrid"):
        # The exact reference. Not in the request path (D-031, F-077) -- kept
        # so Qdrant's approximation can be checked against a known answer.
        vector_store = load_store(args.vectors, MODEL_NAME,
                                  load_model().get_embedding_dimension(),
                                  kind=args.index_kind)
        print(f"faiss: {vector_store.chunk_count:,} chunks ({args.index_kind})")
        dense_scores = vector_search(vector_store, embed_query(args.area), depth)
        rankings.append([article_id for article_id, _ in dense_scores])

    if args.retriever == "dense":
        return dense_scores, records
    # BM25 alone also goes through fuse(), so the printed score column means the
    # same thing in every mode. BM25's raw score is saturated and near-
    # meaningless as an absolute number anyway (F-066).
    return fuse(rankings), records

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL corpus to index")
    parser.add_argument("--area", required=True)
    parser.add_argument("--question", default=None,
                        help="Defaults to a severe-crime question about --area")
    parser.add_argument("--limit", type=int, default=10,
                        help="How many articles to send the model")
    parser.add_argument("--searched-from", default="2021-01-01")
    parser.add_argument("--searched-to", default="2026-08-26")
    parser.add_argument("--today", default=None)
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning:free")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"],
                        default="hybrid",
                        help="F-072: neither is safe alone. Default fuses both.")
    parser.add_argument("--vectors", type=Path, default=Path("processed/vectors"))
    parser.add_argument("--index-kind", choices=["flat", "ivf"], default="flat",
                        help="flat is exact; ivf is approximate and loses results.")
    parser.add_argument("--store", choices=["qdrant", "faiss"], default="qdrant",
                        help="qdrant serves the app; faiss is the exact reference.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what the search returned and stop. No API call.")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Not a file: {args.path}")

    # Article text is non-ASCII; a cp1252 console would raise on it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ranked, records = retrieve(args)
    if not ranked:
        # Not an error and not an empty answer -- the two must stay distinct
        # (CLAUDE.md section 1). Nothing matched, so there is nothing to ask
        # the model about, and no call is spent finding that out.
        raise SystemExit(
            f"\nNothing retrieved for {args.area!r}.\n"
            "That is 'nothing was indexed under this name', which is NOT the "
            "same as 'this area is safe'.")

    ordered_ids = [article_id for article_id, _ in ranked[:args.limit]]
    # Qdrant already returned the text. Only the FAISS path needs the file.
    heads = records or fetch_bodies(args.path, set(ordered_ids))

    print()
    print(f"{'rank':>4}  {'score':>8}  {'article_id':>10}  {'published':<10}  headline")
    print("-" * 110)
    for rank, (article_id, score) in enumerate(ranked[:args.limit], start=1):
        row = heads.get(article_id, {})
        print(f"{rank:>4}  {score:>8.3f}  {article_id:>10}  "
              f"{(row.get('published_at') or '')[:10]:<10}  "
              f"{(row.get('headline') or '')[:60]}")

    if args.dry_run:
        print("\n--dry-run: stopping before the model. No API call spent.")
        return

    # Search order is preserved deliberately: the model reads sources top-down
    # and D-017's block format is written in rank order.
    records = [heads[article_id] for article_id in ordered_ids if article_id in heads]

    missing = len(ordered_ids) - len(records)
    if missing:
        print(f"\nWARNING: {missing} retrieved id(s) not found on a re-read of "
              f"{args.path}. The index and the file disagree.")

    question = args.question or (
        f"What crimes have been reported near {args.area}?")

    print()
    answer = generate(records, args.area, question,
                      args.searched_from, args.searched_to,
                      ModelConfig(model=args.model), today=args.today,
                      force=args.force)

    present(answer, records, args.area, args.searched_from, args.searched_to)

if __name__ == "__main__":
    main()
