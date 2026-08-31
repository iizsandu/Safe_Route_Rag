"""Produce a labelling sheet for the articles retrieval actually returns.

`eval/rohini_labels.csv` labels a RANDOM sample of the pool, which measures the
pool's composition (F-044) and not the ranking. Judging a retriever needs labels
on what it puts in front of a user.

Takes the union of each retriever's top N, so neither is judged on a set chosen
by the other. Writes body text into the sheet because headlines are not enough
to decide -- the assistant mislabelled two articles from headlines alone during
F-070.

Run:
    python -m eval.make_label_sheet processed/rohini_2021.jsonl --area "Rohini, Delhi"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rag.chunk import chunk_records
from rag.embed import embed_passages, embed_query, token_count
from rag.index import build_index, index_text
from rag.search import search

BODY_CHARS = 700

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path)
    parser.add_argument("--area", required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--chunk-tokens", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("eval/rohini_retrieval_sheet.csv"))
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pool = load_jsonl(args.pool)
    by_id = {record["article_id"]: record for record in pool}
    print(f"pool: {len(pool):,} articles")

    index, store = build_index(args.pool, snippet_chars=1)
    results, _ = search(index, args.area, limit=len(pool))
    bm25_rank = {store.get(r.doc_number)["article_id"]: i
                 for i, r in enumerate(results, 1)}

    units, owners = chunk_records(pool, index_text, token_count, args.chunk_tokens)
    print(f"embedding {len(units):,} chunks ...")
    scores = embed_passages(units) @ embed_query(args.area)
    best: dict[int, float] = {}
    for article_id, score in zip(owners, scores):
        best[article_id] = max(best.get(article_id, -2.0), float(score))
    dense_rank = {article_id: i for i, (article_id, _) in
                  enumerate(sorted(best.items(), key=lambda kv: -kv[1]), 1)}

    top_bm25 = {a for a, r in bm25_rank.items() if r <= args.top}
    top_dense = {a for a, r in dense_rank.items() if r <= args.top}
    union = sorted(top_bm25 | top_dense,
                   key=lambda a: min(bm25_rank.get(a, 10**6),
                                     dense_rank.get(a, 10**6)))

    print(f"bm25 top {args.top}: {len(top_bm25)}   "
          f"dense top {args.top}: {len(top_dense)}   "
          f"union: {len(union)}   overlap: {len(top_bm25 & top_dense)}")

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["article_id", "bm25_rank", "dense_rank", "verdict",
                         "why", "headline", "body_head"])
        for article_id in union:
            record = by_id[article_id]
            body = " ".join((record.get("body_text") or "").split())
            writer.writerow([
                article_id,
                bm25_rank.get(article_id, ""),
                dense_rank.get(article_id, ""),
                "", "",
                record.get("headline") or "",
                body[:BODY_CHARS],
            ])
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
