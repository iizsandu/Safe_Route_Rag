"""BM25 vs dense retrieval, same query, same candidate pool.

The comparison F-066 made impossible to duck: BM25 ranks `114156694` below 50 for
"Rohini, Delhi" because the article is headlined *Vijay Vihar*. Lexical matching
cannot know Vijay Vihar is inside Rohini. Whether an embedding does is the single
most valuable thing to learn about dense retrieval on this corpus.

Both retrievers rank the SAME pool so the ranks are directly comparable. Ranking
the full 107k corpus with one and a 790-article slice with the other would
compare two different problems.

Also produces, for free, two measurements PHASE5_PLAN.md asks for before any
full run:

    M1  what fraction of articles exceed the model's 512-token limit
        -> decides chunk-vs-article (D-c)
    M2  embedding throughput, measured rather than guessed
        -> tells us whether 107k articles is 20 minutes or 6 hours

Run:
    python -m eval.compare_retrieval processed/rohini_2021.jsonl --area "Rohini, Delhi"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rag.chunk import chunk_records
from rag.embed import embed_passages, embed_query, load_model, token_count
from rag.index import build_index, index_text
from rag.search import search

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def rank_map(article_ids: list[int]) -> dict[int, int]:
    """article_id -> 1-based rank."""
    return {article_id: position for position, article_id in enumerate(article_ids, 1)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path, help="JSONL candidate pool to rank")
    parser.add_argument("--area", required=True)
    parser.add_argument("--labelled", type=Path,
                        default=Path("eval/rohini_sample20.jsonl"),
                        help="Articles whose ranks we report individually")
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--chunk-tokens", type=int, default=0,
                        help="Chunk size in tokens. 0 embeds whole articles.")
    args = parser.parse_args()

    if not args.pool.is_file():
        raise SystemExit(f"Not a file: {args.pool}")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pool = load_jsonl(args.pool)
    texts = [index_text(record) for record in pool]
    print(f"pool         : {len(pool):,} articles from {args.pool}")

    # --- M1: truncation rate -------------------------------------------------
    limit = load_model().max_seq_length
    counts = np.array([token_count(text) for text in texts])
    over = int((counts > limit).sum())
    print(f"\n--- M1: token lengths (model limit {limit}) ---")
    print(f"  median {int(np.median(counts))}   mean {counts.mean():.0f}   "
          f"max {counts.max()}")
    print(f"  over the limit: {over:,} of {len(counts):,} "
          f"({100 * over / len(counts):.1f}%)  <- silently truncated")

    # --- what actually gets embedded -----------------------------------------
    if args.chunk_tokens:
        units, owners = chunk_records(pool, index_text, token_count,
                                      args.chunk_tokens)
        print(f"\n--- chunked at {args.chunk_tokens} tokens ---")
        print(f"  {len(pool):,} articles  ->  {len(units):,} chunks "
              f"({len(units) / len(pool):.1f} per article)")
    else:
        units, owners = texts, [record["article_id"] for record in pool]

    # --- M2: throughput ------------------------------------------------------
    print(f"\n--- M2: embedding {len(units):,} units ---")
    started = time.perf_counter()
    vectors = embed_passages(units)
    elapsed = time.perf_counter() - started
    print(f"  {elapsed:.1f}s  ->  {len(units) / elapsed:.0f} units/sec")
    print(f"  extrapolated to 107,264 articles: "
          f"{107264 * (len(units) / len(pool)) / (len(units) / elapsed) / 60:.0f} minutes")
    print(f"  vector store: {vectors.nbytes / 1e6:.0f} MB for this pool")

    # --- rank both ways ------------------------------------------------------
    # An article scores as its BEST chunk, never the average. Averaging chunk
    # scores would re-introduce the dilution that chunking exists to remove.
    query = embed_query(args.area)
    unit_scores = vectors @ query
    best: dict[int, float] = {}
    for article_id, score in zip(owners, unit_scores):
        if score > best.get(article_id, -2.0):
            best[article_id] = float(score)

    ranked = sorted(best.items(), key=lambda item: -item[1])
    dense_rank = rank_map([article_id for article_id, _ in ranked])
    by_id = {record["article_id"]: record for record in pool}
    position_of = {record["article_id"]: i for i, record in enumerate(pool)}

    index, store = build_index(args.pool, snippet_chars=1)
    results, _ = search(index, args.area, limit=len(pool))
    bm25_rank = rank_map(
        [store.get(result.doc_number)["article_id"] for result in results])

    print(f"\n--- DENSE top {args.show} for {args.area!r} ---")
    print(f"{'rank':>4}  {'cos':>6}  {'bm25':>5}  {'tokens':>6}  headline")
    print("-" * 104)
    for position, (article_id, score) in enumerate(ranked[:args.show], start=1):
        record = by_id[article_id]
        print(f"{position:>4}  {score:>6.3f}  "
              f"{bm25_rank.get(article_id, 0):>5}  "
              f"{counts[position_of[article_id]]:>6}  "
              f"{(record.get('headline') or '')[:70]}")

    # --- the labelled set, side by side --------------------------------------
    # BM25's ranks are the baseline. A large negative delta is dense retrieval
    # finding something lexical matching could not.
    print(f"\n--- the {args.labelled.name} articles, both rankings ---")
    print(f"{'article_id':>10}  {'bm25':>5}  {'dense':>5}  {'delta':>6}  "
          f"{'tokens':>6}  headline")
    print("-" * 110)
    rows = []
    for record in load_jsonl(args.labelled):
        article_id = record["article_id"]
        if article_id not in dense_rank:
            continue
        bm25 = bm25_rank.get(article_id, len(pool))
        dense = dense_rank[article_id]
        rows.append((dense, bm25, article_id, record.get("headline") or ""))
    for dense, bm25, article_id, headline in sorted(rows):
        print(f"{article_id:>10}  {bm25:>5}  {dense:>5}  {bm25 - dense:>+6}  "
              f"{counts[position_of[article_id]]:>6}  {headline[:60]}")

if __name__ == "__main__":
    main()
