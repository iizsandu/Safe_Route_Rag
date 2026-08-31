"""Turn text into vectors.

**The only module in `rag/` permitted to import a model runtime** (D-026).
Everything else -- ingestion, indexing, BM25 search, the LLM client, generation,
verification, rendering -- still runs in a bare virtualenv with nothing
installed. That boundary is the whole point: it keeps the sparse half
independently testable, and it means one file to rewrite if we later move to
ONNX.

Two things here fail SILENTLY if got wrong -- no exception, no warning, just
quietly worse retrieval:

  1. The query instruction. `bge-en-v1.5` puts an instruction sentence on the
     QUERY only; passages get nothing. This is not the e5 convention
     ("query: " / "passage: "), and applying e5's prefixes to bge degrades
     results with no error. See QUERY_INSTRUCTION.
  2. Pooling. bge uses CLS pooling, not mean pooling. sentence-transformers
     reads this from the model config, which is why D-026 chose it over
     hand-rolled ONNX for the first build.

Run:
    python -m rag.embed
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# Pinned exactly. Embeddings are NOT comparable across models or revisions --
# a vector store built with one model is meaningless to another. `rag/llm.py`
# pins model strings for the same reason.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge-en-v1.5 asks for this on the QUERY side only. Passages are embedded raw.
# Written here once so it cannot be got wrong at each call site.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

@lru_cache(maxsize=1)
def load_model(name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the model once and keep it.

    Loading costs seconds and allocates ~130 MB. Doing it per call would make
    a 107k-article run unusably slow, and would silently reload inside loops.
    """
    return SentenceTransformer(name)

def embed_passages(texts: list[str], name: str = MODEL_NAME) -> np.ndarray:
    """Embed documents. Returns one unit-length row per input text.

    No instruction prefix -- see QUERY_INSTRUCTION. `normalize_embeddings`
    scales every vector to length 1, which makes cosine similarity a plain dot
    product (see `similarity`) and removes one place to make an arithmetic
    mistake later.
    """
    model = load_model(name)
    return model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False)

def embed_query(text: str, name: str = MODEL_NAME) -> np.ndarray:
    """Embed a search query, with the instruction bge expects."""
    model = load_model(name)
    return model.encode([QUERY_INSTRUCTION + text],
                        normalize_embeddings=True,
                        show_progress_bar=False)[0]

def similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity between two vectors from this module.

    Both are unit length already, so the division by magnitudes that cosine
    normally needs is a division by 1. Only valid for vectors produced here
    with normalize_embeddings=True.
    """
    return float(np.dot(left, right))

def token_count(text: str, name: str = MODEL_NAME) -> int:
    """How many tokens this model will see, including its special tokens.

    The number that matters for truncation: anything past `max_seq_length` is
    dropped without warning.
    """
    model = load_model(name)
    return len(model.tokenizer.encode(text))

# --- smoke test -------------------------------------------------------------

PLACES = ["Rohini", "Whitefield", "Koramangala", "Marathahalli",
          "Peelamedu", "Sowripalayam", "Sakthikulangara", "Vytilla"]

# One place, three sentences: the right area, a different area, and an
# unrelated topic. If cosine cannot order these three, dense retrieval cannot
# do the job this product needs (M6 in PHASE5_PLAN.md).
PROBE_QUERY = "crime in Sakthikulangara"
PROBE_TEXTS = [
    "Two men were arrested after a fishing boat owner was assaulted at "
    "Sakthikulangara harbour on Tuesday, police said.",
    "Two men were arrested after a shopkeeper was assaulted in Rohini, "
    "Delhi on Tuesday, police said.",
    "The state cricket association announced its squad for the upcoming "
    "one-day tournament in December.",
]

# The gap BM25 provably cannot close: zero shared words, same event (D-003,
# and the stated purpose of Phase 5).
PARAPHRASE = ("godman released from jail", "Asaram granted bail")

def main() -> None:
    model = load_model()
    print(f"model       : {MODEL_NAME}")
    print(f"dimensions  : {model.get_embedding_dimension()}")
    print(f"max tokens  : {model.max_seq_length}")

    print("\n--- M5: how badly do place names fragment? ---")
    for place in PLACES:
        # -2 removes the [CLS] and [SEP] the tokenizer adds around every input.
        pieces = token_count(place) - 2
        print(f"  {place:<18} {pieces} token(s)")

    print("\n--- M6: can cosine find the right article? ---")
    query = embed_query(PROBE_QUERY)
    labels = ["same place       ", "different place  ", "unrelated topic  "]
    for label, score in zip(labels, embed_passages(PROBE_TEXTS) @ query):
        print(f"  {label} {score:.3f}")

    print("\n--- the gap BM25 cannot close (0 shared words) ---")
    first, second = embed_passages(list(PARAPHRASE))
    print(f"  {PARAPHRASE[0]!r}")
    print(f"  {PARAPHRASE[1]!r}")
    print(f"  similarity {similarity(first, second):.3f}")

if __name__ == "__main__":
    main()
