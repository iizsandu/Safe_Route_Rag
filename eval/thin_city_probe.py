"""Does dense retrieval fail on heavily-fragmented rare place names?

F-068's open question. `Rohini` splits into 2 subword pieces and dense beat BM25
6x on it (F-071). But BM25's advantage is supposed to be *exact matching of rare
proper nouns* -- and the names that fragment badly are the ones where that should
matter: `Sowripalayam` (5 pieces), `Sakthikulangara` (6). In BM25 those are a
single term at idf 9-10, the strongest signal in the index.

No relevance labelling needed. The proxy is mechanical: **does the retrieved
article contain the term at all?** That is a weaker question than "is the
incident there" (F-044 shows the two differ), but it is exactly the question
F-068 raised -- can dense retrieval even FIND documents about a fragmented name.

Pool = every article containing the term, plus a random sample of articles that
do not, so there is something to be wrong about.

Run:
    python -m eval.thin_city_probe processed/recent_2021.jsonl --term peelamedu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from rag.chunk import chunk_records
from rag.embed import embed_passages, embed_query, token_count
from rag.index import build_index, index_text
from rag.search import search

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--term", required=True, help="Place name, lowercase")
    parser.add_argument("--query", default=None, help="Defaults to --term")
    parser.add_argument("--distractors", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-tokens", type=int, default=200)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(args.seed)
    term = args.term.lower()
    query = args.query or args.term

    # Reservoir sampling for the distractors: one pass, constant memory (D-002).
    targets: list[dict] = []
    others: list[dict] = []
    seen_others = 0
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            haystack = ((record.get("headline") or "") + " " +
                        (record.get("body_text") or "")).lower()
            if term in haystack:
                targets.append(record)
                continue
            seen_others += 1
            if len(others) < args.distractors:
                others.append(record)
            else:
                slot = random.randrange(seen_others)
                if slot < args.distractors:
                    others[slot] = record

    pool = targets + others
    random.shuffle(pool)
    target_ids = {record["article_id"] for record in targets}
    print(f"term        : {term!r}")
    print(f"pool        : {len(pool):,}  "
          f"({len(targets)} contain the term, {len(others)} do not)")
    if not targets:
        raise SystemExit("No article contains that term.")

    pool_path = Path("eval/_probe_pool.jsonl")
    with pool_path.open("w", encoding="utf-8") as handle:
        for record in pool:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    pieces = token_count(args.term) - 2
    print(f"tokenizes to: {pieces} piece(s)")

    index, store = build_index(pool_path, snippet_chars=1)
    results, _ = search(index, query, limit=len(pool))
    bm25_ids = [store.get(r.doc_number)["article_id"] for r in results]

    units, owners = chunk_records(pool, index_text, token_count, args.chunk_tokens)
    print(f"embedding {len(units):,} chunks ...")
    scores = embed_passages(units) @ embed_query(query)
    best: dict[int, float] = {}
    for article_id, score in zip(owners, scores):
        best[article_id] = max(best.get(article_id, -2.0), float(score))
    dense_ids = [a for a, _ in sorted(best.items(), key=lambda kv: -kv[1])]

    print(f"\nhow many of the top k actually contain {term!r}?")
    print(f"{'':8}{'@5':>8}{'@10':>8}{'@20':>8}{'@30':>8}")
    for name, ids in (("BM25", bm25_ids), ("DENSE", dense_ids)):
        row = "".join(
            f"{sum(1 for a in ids[:k] if a in target_ids) / k:>7.0%} "
            for k in (5, 10, 20, 30))
        print(f"{name:8}{row}")

    pool_path.unlink()

if __name__ == "__main__":
    main()
