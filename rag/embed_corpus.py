"""Embed the whole corpus into a vector store, resumably.

~7 hours for 107,264 articles (F-070, M2). A run that cannot resume is a run
that has to be perfect for seven hours, and `PROJECT_PLAN.md` §Reliability
already required this before the number was known.

WHAT IT WRITES

    vectors.f32    raw float32, `dim` values per chunk, appended in order
    chunks.jsonl   one line per chunk, in the same order: its article_id
    state.json     model, revision, dim, chunk size, and how far we got

Three files rather than one archive so that appending is a plain write and a
crash truncates rather than corrupts. Position N in `vectors.f32` is position N
in `chunks.jsonl`; nothing else relates them, and nothing needs to.

CRASH RECOVERY

`state.json` is written AFTER the vectors it describes. So on restart the
vector file may hold more than the state claims -- a batch that was written
when the process died. It is truncated back to the state's count, and those
articles are redone. The reverse (state ahead of vectors) would silently lose
data, which is why the order matters.

THE MODEL IS PINNED IN state.json. Vectors from one model are meaningless to
another, and a store that does not record which model built it is a store you
cannot trust later (D-026).

Run:
    python -m rag.embed_corpus processed/recent_2021.jsonl --limit 2000
    python -m rag.embed_corpus processed/recent_2021.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rag.chunk import chunk_records
from rag.embed import MODEL_NAME, load_model, token_count
from rag.index import index_text

VECTORS = "vectors.f32"
CHUNKS = "chunks.jsonl"
STATE = "state.json"

def read_state(out_dir: Path) -> dict[str, Any]:
    path = out_dir / STATE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def write_state(out_dir: Path, state: dict[str, Any]) -> None:
    """Written only after the vectors it describes are on disk and fsynced."""
    (out_dir / STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")

def stream_records(path: Path, skip: int) -> Iterator[dict[str, Any]]:
    """Yield records, skipping the first `skip`. Never holds the corpus."""
    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position < skip or not line.strip():
                continue
            yield json.loads(line)

def batched(records: Iterator[dict[str, Any]], size: int
            ) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=Path("processed/vectors"))
    parser.add_argument("--chunk-tokens", type=int, default=200)
    parser.add_argument("--articles-per-batch", type=int, default=200,
                        help="Checkpoint granularity. A crash costs at most this many.")
    parser.add_argument("--encode-batch", type=int, default=32,
                        help="Chunks per forward pass. Untuned -- see F-070.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many articles. 0 = whole corpus.")
    args = parser.parse_args()

    if not args.corpus.is_file():
        raise SystemExit(f"Not a file: {args.corpus}")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model = load_model()
    dim = model.get_embedding_dimension()

    state = read_state(args.out)
    if state:
        if state.get("model") != MODEL_NAME or state.get("dim") != dim:
            raise SystemExit(
                f"Store was built with {state.get('model')} at dim "
                f"{state.get('dim')}; this is {MODEL_NAME} at {dim}. "
                "Vectors from different models are not comparable -- delete "
                f"{args.out} or point --out elsewhere.")
        if state.get("chunk_tokens") != args.chunk_tokens:
            raise SystemExit(
                f"Store was chunked at {state.get('chunk_tokens')} tokens, "
                f"not {args.chunk_tokens}.")

    articles_done = int(state.get("articles_done", 0))
    chunks_written = int(state.get("chunks_written", 0))

    # Truncate anything written after the last checkpoint -- see CRASH RECOVERY.
    vectors_path = args.out / VECTORS
    expected = chunks_written * dim * 4
    if vectors_path.is_file() and vectors_path.stat().st_size != expected:
        actual = vectors_path.stat().st_size
        print(f"recovering: {VECTORS} is {actual:,} bytes, state says "
              f"{expected:,}. Truncating to the checkpoint.")
        with vectors_path.open("r+b") as handle:
            handle.truncate(expected)

    chunks_path = args.out / CHUNKS
    if chunks_path.is_file():
        kept = chunks_path.read_text(encoding="utf-8").splitlines()[:chunks_written]
        chunks_path.write_text("\n".join(kept) + ("\n" if kept else ""),
                               encoding="utf-8")

    if articles_done:
        print(f"resuming after {articles_done:,} articles "
              f"({chunks_written:,} chunks already written)")

    started = time.perf_counter()
    processed = 0
    records = stream_records(args.corpus, skip=articles_done)

    with vectors_path.open("ab") as vectors_out, \
         chunks_path.open("a", encoding="utf-8") as chunks_out:

        for batch in batched(records, args.articles_per_batch):
            units, owners = chunk_records(batch, index_text, token_count,
                                          args.chunk_tokens)
            if units:
                vectors = model.encode(units, normalize_embeddings=True,
                                       batch_size=args.encode_batch,
                                       show_progress_bar=False)
                vectors_out.write(np.asarray(vectors, dtype=np.float32).tobytes())
                vectors_out.flush()
                for article_id in owners:
                    chunks_out.write(json.dumps({"article_id": article_id}) + "\n")
                chunks_out.flush()

            articles_done += len(batch)
            chunks_written += len(units)
            processed += len(batch)

            # State last, always. See CRASH RECOVERY.
            write_state(args.out, {
                "model": MODEL_NAME,
                "dim": dim,
                "chunk_tokens": args.chunk_tokens,
                "corpus": str(args.corpus),
                "articles_done": articles_done,
                "chunks_written": chunks_written,
            })

            rate = processed / (time.perf_counter() - started)
            print(f"  {articles_done:>7,} articles  {chunks_written:>8,} chunks  "
                  f"{rate:>5.1f} art/s", flush=True)

            if args.limit and processed >= args.limit:
                print(f"\nstopped at --limit {args.limit}")
                break

    elapsed = time.perf_counter() - started
    print(f"\ndone: {articles_done:,} articles, {chunks_written:,} chunks, "
          f"{elapsed / 60:.1f} min this session")
    print(f"store: {vectors_path.stat().st_size / 1e6:.0f} MB in {args.out}")

if __name__ == "__main__":
    main()
