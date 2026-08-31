"""Load the chunk vectors into Qdrant; article text into DynamoDB (F-079).

D-031: the app must hold no data. The vectors, BM25 weights and the dates
to filter on live in Qdrant; the article text lives in DynamoDB, fetched
by rag.articles -- not copied into every chunk of the same article.

Run once. Takes a few minutes.

    python -m rag.qdrant_load processed/vectors processed/recent_2021.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from qdrant_client import QdrantClient, models

from rag.articles import connect as connect_articles, put_article
from rag.index import build_index
from rag.search import B, K1, inverse_document_frequency
from rag.tokenizer import tokenize

COLLECTION = "safe_route_chunks"
BATCH = 256

def term_id(term: str) -> int:
    """A stable integer for a term.

    crc32, NOT Python's hash(): hash() is randomised per process, so the same
    word would get a different id on every run and the stored index would
    silently stop matching the queries made against it.
    """
    return zlib.crc32(term.encode("utf-8"))

def bm25_sparse(text: str, idf: dict[str, float], avg_length: float
                ) -> models.SparseVector:
    """One article as BM25 term weights.

    Qdrant's own sparse scoring applies IDF and nothing else. Our BM25 also
    has term-frequency saturation (k1) and document-length normalisation (b),
    and those two are exactly what F-043 and F-047 measured on this corpus.
    Using Qdrant's default would silently change the scoring that F-071 and
    F-073 were measured with.

    So the full BM25 weight is computed here and stored as the value. The
    query sends 1.0 per term, so Qdrant's dot product returns the BM25 score
    unchanged -- same numbers, different machine.
    """
    counts = Counter(tokenize(text))
    length = sum(counts.values())
    normalizer = 1 - B + B * (length / avg_length if avg_length else 1.0)

    indices, values = [], []
    for term, frequency in counts.items():
        weight = idf.get(term)
        if weight is None:
            continue
        indices.append(term_id(term))
        values.append(weight * (frequency * (K1 + 1)) /
                      (frequency + K1 * normalizer))
    return models.SparseVector(indices=indices, values=values)

def corpus_statistics(path: Path) -> tuple[dict[str, float], float]:
    """IDF per term, and the average document length.

    Reuses rag.index rather than recomputing: the numbers must match what
    rag/search.py produces, or the two retrievers stop being comparable.
    """
    index, _ = build_index(path, snippet_chars=1)
    idf = {term: inverse_document_frequency(index.doc_count, len(postings))
           for term, postings in index.postings.items()}
    return idf, index.avg_doc_length

def load_articles(path: Path) -> dict[int, dict[str, Any]]:
    """article_id -> the fields a chunk needs to carry.

    Held in memory for the load only. The SERVER never does this -- that is
    the whole point of moving the text into Qdrant (D-031).
    """
    out: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            published = (record.get("published_at") or "")[:10]
            out[record["article_id"]] = {
                "headline": record.get("headline") or "",
                "body_text": record.get("body_text") or "",
                # A citation the reader cannot open is a weak citation
                # (CLAUDE.md section 9). The URL travels with the article.
                "url": record.get("url") or "",
                "published_at": published,
                # An integer date, because range filters on strings are a
                # trap. 2026-08-29 -> 20260829. Sorts and compares correctly.
                "published_ts": int(published.replace("-", "") or 0),
            }
    return out

def points(vectors: np.ndarray, owners: list[int],
           articles: dict[int, dict[str, Any]],
           idf: dict[str, float], avg_length: float
           ) -> Iterator[models.PointStruct]:
    """One point per chunk: dense vector, BM25 weights, and the article text.

    The BM25 weights are built from the ARTICLE, not the chunk -- that is what
    rag/search.py indexes, and what F-071's 10% and F-073's 100% were measured
    against. Chunk-level keyword search might be better; it is unmeasured, so
    it is not being changed here.

    Every chunk of an article therefore carries the same sparse vector.
    Redundant, and accepted: keeping the measured baseline intact is worth
    more right now than the storage.
    """
    sparse_cache: dict[int, models.SparseVector] = {}
    for index, (vector, article_id) in enumerate(zip(vectors, owners)):
        article = articles.get(article_id)
        if article is None:
            continue
        if article_id not in sparse_cache:
            sparse_cache[article_id] = bm25_sparse(
                article["headline"] + " " + article["body_text"],
                idf, avg_length)
        yield models.PointStruct(
            id=index,
            vector={"dense": vector.tolist(), "text": sparse_cache[article_id]},
            # body_text deliberately excluded -- it lives in DynamoDB now
            # (F-079), not copied into every chunk of the same article.
            payload={
                "article_id": article_id,
                "headline": article["headline"],
                "url": article["url"],
                "published_at": article["published_at"],
                "published_ts": article["published_ts"],
            },
        )

def batched(items: Iterator[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vectors", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    meta = json.loads((args.vectors / "state.json").read_text(encoding="utf-8"))
    dim = int(meta["dim"])
    raw = np.fromfile(args.vectors / "vectors.f32", dtype=np.float32)
    vectors = raw.reshape(-1, dim)
    owners = [json.loads(line)["article_id"]
              for line in (args.vectors / "chunks.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(vectors):,} vectors, dim {dim}")

    print(f"reading {args.corpus} ...")
    articles = load_articles(args.corpus)
    print(f"{len(articles):,} articles")

    print("writing article bodies to DynamoDB ...")
    table = connect_articles()
    for done, (article_id, article) in enumerate(articles.items(), start=1):
        put_article(table, article_id, article["body_text"])
        if done % 5_000 == 0:
            print(f"  {done:,} / {len(articles):,}")
    print(f"done: {len(articles):,} article bodies written")

    print("building corpus statistics for BM25 ...")
    idf, avg_length = corpus_statistics(args.corpus)
    print(f"{len(idf):,} terms, avg length {avg_length:.1f}")

    client = QdrantClient(url=args.url)
    client.recreate_collection(
        collection_name=COLLECTION,
        # Vectors are unit length already (rag/embed.py), so cosine and dot
        # agree. Cosine is stated because it is what we mean.
        vectors_config={
            "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        },
        # No modifier. The stored values are already full BM25 weights --
        # IDF, saturation and length normalisation are baked in by
        # bm25_sparse(), so Qdrant only has to do the dot product. Using
        # Modifier.IDF here would apply IDF a second time.
        sparse_vectors_config={"text": models.SparseVectorParams()},
    )
    # Without this index a date filter scans everything. This is the thing
    # FAISS could not do at all.
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="published_ts",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    done = 0
    for batch in batched(points(vectors, owners, articles, idf, avg_length), BATCH):
        client.upsert(collection_name=COLLECTION, points=batch, wait=False)
        done += len(batch)
        if done % 25_600 == 0:
            print(f"  {done:,} / {len(vectors):,}")
    print(f"done: {done:,} points")

    print(client.get_collection(COLLECTION).points_count, "points in Qdrant")

if __name__ == "__main__":
    main()
