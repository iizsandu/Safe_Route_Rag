"""Search Qdrant: keyword and meaning together, in one request.

D-031: the app holds no data. Qdrant holds the vectors and the BM25 weights;
DynamoDB holds the article text (F-079); this module does the fusion and
brings the two together.

Replaces three things the pipeline used to do:
    - rebuilding the BM25 index on every run (~60 seconds)
    - re-reading the 292 MB corpus file to get article text
    - fusing two rankings in our own code
"""

from __future__ import annotations

import zlib
from collections import Counter
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from rag.articles import get_bodies
from rag.tokenizer import tokenize

COLLECTION = "safe_route_chunks"

# Ask Qdrant for more CHUNKS than the ARTICLES we want: several chunks can
# belong to one article, so 10 chunks might be only 3 distinct articles.
CHUNK_OVERSAMPLE = 30

def connect(url: str = "http://localhost:6333", timeout: int = 60) -> QdrantClient:
    """The default client timeout is a few seconds, which a FILTERED search can
    exceed: with a date filter Qdrant has to walk further through the graph to
    find enough points that pass, so the first filtered query is much slower
    than an unfiltered one."""
    return QdrantClient(url=url, timeout=timeout)

def sparse_query(text: str) -> models.SparseVector:
    """The query as term ids with weight 1.

    The stored values are already full BM25 weights (rag/qdrant_load.py), so
    a query of 1.0 per term makes Qdrant's dot product return exactly the BM25
    score rag/search.py would have computed.

    crc32 must match the loader's term_id() or nothing matches at all.
    """
    counts = Counter(tokenize(text))
    return models.SparseVector(
        indices=[zlib.crc32(term.encode("utf-8")) for term in counts],
        values=[1.0] * len(counts),
    )

def _one_search(client: QdrantClient, query: Any, using: str, depth: int,
                query_filter: models.Filter | None
                ) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """One retriever's ranking, plus the article text it returned.

    Chunks are collapsed to articles here, keeping each article's BEST chunk.
    Averaging would re-introduce the dilution chunking exists to remove (F-070).
    """
    hits = client.query_points(
        collection_name=COLLECTION, query=query, using=using,
        limit=depth, query_filter=query_filter, with_payload=True,
    ).points

    best: dict[int, float] = {}
    payloads: dict[int, dict[str, Any]] = {}
    for hit in hits:
        article_id = int(hit.payload["article_id"])
        if hit.score > best.get(article_id, -1e9):
            best[article_id] = float(hit.score)
            payloads[article_id] = hit.payload

    ranking = [article_id for article_id, _ in
               sorted(best.items(), key=lambda item: -item[1])]
    return ranking, payloads

def date_bound(text: str | None) -> int | None:
    """"2021-01-01" -> 20210101. None if absent or malformed.

    An integer, because range filters on date STRINGS are a trap -- they
    compare lexically, and "2021-1-1" sorts after "2021-01-01".
    """
    if not text:
        return None
    digits = str(text).replace("-", "").strip()
    return int(digits) if digits.isdigit() and len(digits) == 8 else None

def search(
    client: QdrantClient,
    articles_table: Any,
    query_vector: np.ndarray,
    query_text: str,
    limit: int,
    since: int | None = None,
    until: int | None = None,
    fuse=None,
) -> list[tuple[int, float, dict[str, Any]]]:
    """Hybrid search. Returns (article_id, score, payload) best first.

    TWO requests, not one, and the caller's `fuse` combines them.

    Qdrant can fuse internally with `FusionQuery(Fusion.RRF)`, and that was
    tried first. It uses a much smaller RRF constant than our k=60, so an
    article ranked highly by ONE retriever outranks an article both retrievers
    agree on -- and two junk results (a storm story, a judge retiring) returned
    to the Rohini top 10 that our fusion had removed. k=60 is what F-073's
    10-of-10 was measured with, so the fusion stays ours.

    Two round trips to a service on the same machine cost microseconds. The
    expensive things -- the 60-second index rebuild and the 292 MB file read --
    are gone either way.

    `since` is an integer date like 20260101 -- the filter FAISS could not do
    at all, and the reason --searched-from can finally mean something.
    """
    depth = limit * CHUNK_OVERSAMPLE

    # The filter FAISS could not do at all. Applied INSIDE both searches, so
    # the top 10 are the best 10 IN THE WINDOW -- filtering afterwards would
    # return 10 results and then throw most of them away.
    query_filter = None
    if since is not None or until is not None:
        query_filter = models.Filter(must=[models.FieldCondition(
            key="published_ts",
            range=models.Range(gte=since, lte=until))])

    dense_ranking, dense_payloads = _one_search(
        client, query_vector.tolist(), "dense", depth, query_filter)
    sparse_ranking, sparse_payloads = _one_search(
        client, sparse_query(query_text), "text", depth, query_filter)

    payloads = {**sparse_payloads, **dense_payloads}
    ranked = fuse([dense_ranking, sparse_ranking])
    final = [(article_id, score, payloads[article_id])
             for article_id, score in ranked[:limit]
             if article_id in payloads]

    # Fetched for exactly the articles that survived fusion -- not the
    # oversampled `depth` set Qdrant was asked for (CHUNK_OVERSAMPLE=30).
    bodies = get_bodies(articles_table, [article_id for article_id, _, _ in final])
    for article_id, _, payload in final:
        payload["body_text"] = bodies.get(article_id, "")
    return final
