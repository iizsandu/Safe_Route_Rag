"""Cut an article into pieces small enough to embed without truncation.

Two problems this solves, measured:

  TRUNCATION  39.2% of articles exceed the model's 512-token limit and lose
              everything past it, silently (F-070, M1).
  DILUTION    pooling averages every token into one vector, so a place name
              mentioned once in a 300-token article is ~2% of the result. The
              worst dense miss so far was an article whose Rohini link was
              incidental and buried -- rank 745 of 790 (F-070).

**No model runtime is imported here.** Token counting arrives as a function
argument, so this module runs and is testable in a bare virtualenv (D-026).

Sentences are kept whole. A chunk that ends mid-sentence is harder to read,
and a chunk you cannot read is a chunk you cannot debug (CLAUDE.md section 7).
"""

from __future__ import annotations

import re
from typing import Callable

# News prose, so this is approximate: "Mr. Sharma" and "Rs. 20 lakh" will split
# wrongly. Accepted -- a mis-split sentence lands one clause in a neighbouring
# chunk, which costs a little context, not correctness. A real sentence
# segmenter would be a dependency (D-026 keeps this module clean).
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    """Split on sentence endings, dropping empties."""
    return [part.strip() for part in SENTENCE_END.split(text) if part.strip()]

def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 200,
    overlap_sentences: int = 1,
) -> list[str]:
    """Cut `text` into chunks of at most `max_tokens`, splitting on sentences.

    `max_tokens` is 200 rather than the model's 512 on purpose. Using the full
    window would barely change anything for the median article (446 tokens)
    and would leave dilution untouched -- and dilution, not truncation, is the
    stronger reason to chunk.

    `overlap_sentences` repeats the tail of each chunk at the head of the next,
    so an incident described across a chunk boundary appears whole in at least
    one chunk. Without it, a sentence pair split down the middle is in neither.

    A single sentence longer than `max_tokens` is emitted alone; the model will
    truncate it. Rare, and better than dropping it.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = count_tokens(sentence)

        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(count_tokens(part) for part in current)

        current.append(sentence)
        current_tokens += tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_records(
    records: list[dict],
    text_of: Callable[[dict], str],
    count_tokens: Callable[[str], int],
    max_tokens: int = 200,
) -> tuple[list[str], list[int]]:
    """Chunk many records, keeping each chunk's article_id alongside it.

    Returns (chunk_texts, article_ids) as two parallel lists rather than a list
    of pairs, because the embedder wants a flat list of strings and the ranker
    wants to map results back. Losing this mapping would make a retrieved chunk
    uncitable, and every claim must trace to an article (CLAUDE.md section 9).
    """
    texts: list[str] = []
    owners: list[int] = []
    for record in records:
        for chunk in chunk_text(text_of(record), count_tokens, max_tokens):
            texts.append(chunk)
            owners.append(record["article_id"])
    return texts, owners
