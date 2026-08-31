"""One area in, one verified answer out.

Shared by the CLI (`rag.pipeline`) and the web app (`rag.app`) so the
retrieve -> generate -> verify -> repair -> render sequence exists exactly
once. Two copies would drift, and the way they would drift is one of them
rendering without verifying (D-018).

Holds no data (D-031). The model and the Qdrant connection are passed in,
created once by whoever is serving, never per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from rag.embed import embed_query
from rag.generate import generate
from rag.llm import ModelConfig
from rag.qdrant_search import date_bound, search as qdrant_search
from rag.render import render
from rag.render_html import render_html
from rag.verify import ERROR, Finding, repair, verify

RRF_K = 60

def fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: combine rankings by POSITION, never by score.

    BM25 runs 0-15 and cosine runs 0.4-0.9. They measure different things on
    different scales, so adding them is meaningless -- and normalising first
    only hides the assumption that the scales are comparable.

    k=60 is from the original paper, and is what F-073's 10-of-10 was measured
    with. Qdrant can fuse internally but uses a far smaller constant, which let
    two junk results back into the Rohini top 10: with a small k, one
    retriever's confident wrong answer beats an article both retrievers agree
    on. The constant is doing real work, so the fusion stays here.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, article_id in enumerate(ranking, start=1):
            scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: -item[1])

@dataclass
class ServiceAnswer:
    area: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    text: str | None = None          # None when verification blocked it
    html: str | None = None          # the same answer, for a browser
    cached: bool = False

    @property
    def blocked(self) -> bool:
        return any(f.severity == ERROR for f in self.findings)

def answer(
    area: str,
    client: Any,
    articles_table: Any,
    config: ModelConfig,
    cache_dir: Path,
    *,
    question: str | None = None,
    limit: int = 10,
    searched_from: str = "2021-01-01",
    searched_to: str = "2026-08-29",
    today: str | None = None,
    force: bool = False,
) -> ServiceAnswer:
    """Retrieve, generate, verify, repair, render.

    Returns the answer even when it fails verification -- `text` is None and
    `findings` says why. Deciding what a user sees is the caller's job; this
    only refuses to render something unsafe.
    """
    hits = qdrant_search(
        client, articles_table, embed_query(area), area, limit,
        since=date_bound(searched_from), until=date_bound(searched_to),
        fuse=fuse,
    )
    records = [payload for _, _, payload in hits]
    result = ServiceAnswer(area=area, sources=records)
    if not records:
        # Not an error and not an empty answer -- the two must stay distinct
        # (CLAUDE.md section 1). Nothing matched, so no call is spent finding
        # that out, and the caller must not print this as reassurance.
        return result

    generated = generate(
        records, area,
        question or f"What crimes have been reported near {area}?",
        searched_from, searched_to, config, cache_dir,
        today=today, force=force,
    )
    result.incidents = generated.incidents
    result.cached = generated.cached
    result.findings = repair(generated, verify(generated, records))

    if not result.blocked:
        window = (date(*map(int, searched_from.split("-"))),
                  date(*map(int, searched_to.split("-"))))
        # Two renderings of the same facts: text for a terminal, HTML for a
        # reader. Both are ours, never the model's (D-018).
        result.text = render(generated.incidents, area, *window, records)
        result.html = render_html(generated.incidents, area, *window, records)
    return result
