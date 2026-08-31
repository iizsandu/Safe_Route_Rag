"""Build and search the FAISS index over the chunk vectors (D-028).

Two index types, and the second is only trustworthy because of the first:

    flat   IndexFlatIP -- EXACT brute force. Slow, and the ground truth.
    ivf    IndexIVFFlat -- approximate. Fast, and loses results.

`eval/faiss_recall.py` measures what `ivf` loses against `flat`. Keeping the
exact index is the whole reason that number can exist: F-067 U2 records that
this project cannot measure retrieval recall against reality, but it CAN
measure an approximation against an exact answer, because the exact answer is
computable. An approximation whose error is measured is engineering; one whose
error is assumed is a guess that happens to be fast.

Vectors are unit length (`rag/embed.py`), so inner product IS cosine and no
extra normalisation step is needed.

An article scores as its BEST chunk, never the average -- averaging would
re-introduce the dilution chunking exists to remove (F-070).

Run:
    python -m rag.vectors processed/vectors --kind flat
    python -m rag.vectors processed/vectors --kind ivf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# How many CHUNKS to pull back per article we want. An article's best chunk can
# sit well below the top of the chunk ranking when many articles each have one
# strong chunk, so asking for exactly `limit` chunks would return far fewer
# than `limit` distinct articles.
CHUNK_OVERSAMPLE = 30

@dataclass(frozen=True)
class VectorStore:
    index: faiss.Index
    article_ids: np.ndarray  # article_ids[i] owns vector i
    meta: dict[str, Any]

    @property
    def chunk_count(self) -> int:
        return len(self.article_ids)

def read_raw(directory: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load vectors.f32 + chunks.jsonl + state.json, checking they agree."""
    meta = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    dim = int(meta["dim"])

    raw = np.fromfile(directory / "vectors.f32", dtype=np.float32)
    if raw.size % dim:
        raise SystemExit(
            f"vectors.f32 holds {raw.size:,} floats, not a multiple of {dim}. "
            "The store is truncated -- re-run rag.embed_corpus to repair it.")
    vectors = raw.reshape(-1, dim)

    article_ids = np.array(
        [json.loads(line)["article_id"]
         for line in (directory / "chunks.jsonl").read_text(
             encoding="utf-8").splitlines() if line.strip()],
        dtype=np.int64,
    )
    if len(vectors) != len(article_ids):
        raise SystemExit(
            f"{len(vectors):,} vectors but {len(article_ids):,} chunk ids. "
            "The two files disagree; re-run rag.embed_corpus.")
    return vectors, article_ids, meta

def build(directory: Path, kind: str = "flat", nlist: int = 0) -> Path:
    """Build a FAISS index from the raw vectors and write it beside them."""
    vectors, _, meta = read_raw(directory)
    dim = int(meta["dim"])
    print(f"{len(vectors):,} vectors, dim {dim}, kind={kind}")

    if kind == "flat":
        # IndexFlatIP stores every vector and scans all of them. No training,
        # no parameters, no approximation -- the answer is the true answer.
        index = faiss.IndexFlatIP(dim)
    elif kind == "ivf":
        # IVF partitions the space into `nlist` cells around centroids found by
        # k-means, then searches only the `nprobe` cells nearest the query. That
        # is where both the speed and the lost results come from: a true
        # neighbour sitting in an unsearched cell is simply never seen.
        # sqrt(n) cells is the usual starting point -- ~600 here.
        nlist = nlist or int(np.sqrt(len(vectors)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)
        print(f"training k-means on {nlist} cells ...")
        index.train(vectors)
    else:
        raise SystemExit(f"Unknown kind: {kind}")

    index.add(vectors)
    out = directory / f"index-{kind}.faiss"
    faiss.write_index(index, str(out))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return out

def load_store(directory: Path, expect_model: str, expect_dim: int,
               kind: str = "flat", nprobe: int = 16) -> VectorStore:
    """Load an index, refusing one built by a different model.

    Vectors from two models are not comparable, and mixing them produces no
    error -- only silently wrong results. The model is recorded at write time
    (`rag/embed_corpus.py`) precisely so this check can exist.
    """
    meta = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if meta.get("model") != expect_model or meta.get("dim") != expect_dim:
        raise SystemExit(
            f"{directory} was built with {meta.get('model')} at dim "
            f"{meta.get('dim')}; this process uses {expect_model} at "
            f"{expect_dim}. Refusing to search it.")

    path = directory / f"index-{kind}.faiss"
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist. Build it first:\n"
            f"    python -m rag.vectors {directory} --kind {kind}")

    index = faiss.read_index(str(path))
    if hasattr(index, "nprobe"):
        # How many cells to search. Higher = closer to exact, and slower.
        index.nprobe = nprobe

    article_ids = np.array(
        [json.loads(line)["article_id"]
         for line in (directory / "chunks.jsonl").read_text(
             encoding="utf-8").splitlines() if line.strip()],
        dtype=np.int64,
    )
    return VectorStore(index=index, article_ids=article_ids, meta=meta)

def search(store: VectorStore, query: np.ndarray, limit: int
           ) -> list[tuple[int, float]]:
    """Return (article_id, score) best first, scoring each article by its best chunk."""
    want = min(limit * CHUNK_OVERSAMPLE, store.chunk_count)
    scores, positions = store.index.search(
        np.asarray([query], dtype=np.float32), want)

    best: dict[int, float] = {}
    for position, score in zip(positions[0], scores[0]):
        if position < 0:          # FAISS pads with -1 when it finds too few
            continue
        article_id = int(store.article_ids[position])
        if score > best.get(article_id, -2.0):
            best[article_id] = float(score)

    return sorted(best.items(), key=lambda item: -item[1])[:limit]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kind", choices=["flat", "ivf"], default="flat")
    parser.add_argument("--nlist", type=int, default=0,
                        help="IVF cells. 0 = sqrt(n).")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build(args.directory, args.kind, args.nlist)

if __name__ == "__main__":
    main()
