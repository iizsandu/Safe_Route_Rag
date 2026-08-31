"""Turn what a user typed into something the retriever can use.

The stage `ARCHITECTURE.md` has always called TERMS and never had. Until now
the raw text WAS the query: typing "is it safe in Rohini at night" sent that
whole sentence to BM25, and pasted it into the prompt where a place name
belonged.

One model call. Structured output, checked in code, same shape as everything
else here (D-018).

THREE THINGS IT MUST NOT DO
  * invent a city. "Rohini" alone retrieves Mumbai articles about a WOMAN
    named Rohini (F-066). Guessing "Delhi" would give a confident answer about
    a place the user did not mean, and this is a safety product -- so it asks
    instead.
  * invent a date. Absent means the full corpus window, never a guess.
  * answer the question. It only decides WHERE and WHEN to look.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from rag.generate import extract_json
from rag.llm import ModelConfig, complete

INSTRUCTIONS = """You turn a user's question into search parameters for a crime-news
archive. You do NOT answer the question.

TODAY IS {today}. The archive covers {corpus_from} to {corpus_to}.

Return ONLY this JSON object:

{{"places": ["Area, City"],
  "from": "YYYY-MM-DD" or null,
  "to": "YYYY-MM-DD" or null,
  "needs_city": ["Area"],
  "asking_about": "place_safety" | "a_person" | "something_else",
  "unclear": false}}

RULES

1. places -- every location the user named, each as "Area, City".
   A route ("Rohini to Saket") is TWO places, both listed.

2. NEVER invent a city. If the user names an area without a city, put the
   area in `needs_city` and NOT in `places`. A wrong city returns confident
   results about somewhere the user did not mean.
   "Rohini, Delhi"  -> places
   "Rohini"         -> needs_city

3. from / to -- only if the user gave a time. "last 6 months" counts,
   "recently" does not. Use null when they gave none. Never guess.

4. asking_about -- what the user WANTS.

   "place_safety"    crime, safety or incidents at a place.
                     "is it safe around X", "crime in X",
                     "what happened in X", "X to Y at night".

   "a_person"        the question is about a named individual.
                     "is Rajesh Kumar a criminal", "was my neighbour
                     arrested". Choose this even if a place is also named.

   "something_else"  anything else -- cafes, directions, weather,
                     arithmetic, recipes.

   A place name does NOT make it "place_safety".
   "cafe near Rohini, Delhi" is "something_else".

5. unclear: true when the user named no place at all. Then places and
   needs_city are both empty.

6. Do not answer the question, summarise, or add commentary.

USER QUESTION:
"""

# A closed set, like LEGAL_STATUS in verify.py. A field that accepts arbitrary
# values cannot be branched on safely -- anything outside this set is treated
# as a refusal, never as permission to proceed.
ASKING_ABOUT = {"place_safety", "a_person", "something_else"}


@dataclass
class Understanding:
    places: list[str] = field(default_factory=list)
    needs_city: list[str] = field(default_factory=list)
    since: str | None = None
    until: str | None = None
    unclear: bool = False
    asking_about: str = "something_else"
    cached: bool = False

    @property
    def ready(self) -> bool:
        """True only when there is somewhere to search."""
        return bool(self.places) and not self.unclear

def understand(
    text: str,
    config: ModelConfig,
    cache_dir: Path,
    corpus_from: str = "2021-01-01",
    corpus_to: str = "2026-08-29",
    today: str | None = None,
    force: bool = False,
) -> Understanding:
    """One call: a user's sentence in, search parameters out."""
    prompt = INSTRUCTIONS.format(
        today=today or date.today().isoformat(),
        corpus_from=corpus_from, corpus_to=corpus_to,
    ) + text.strip()

    reply = complete(prompt, config, cache_dir, label="understand", force=force)
    parsed: dict[str, Any] = extract_json(reply.text, require="places")

    # Everything below is defensive on purpose. This runs before retrieval, so
    # a malformed field here becomes a wrong search rather than a visible
    # error -- the failure mode that is hardest to notice.
    def strings(key: str) -> list[str]:
        value = parsed.get(key)
        return [str(v).strip() for v in value if str(v).strip()] \
            if isinstance(value, list) else []

    def day(key: str) -> str | None:
        value = parsed.get(key)
        text_value = str(value).strip() if value else ""
        return text_value if len(text_value) == 10 and text_value[4] == "-" else None

    places = strings("places")
    needs_city = strings("needs_city")
    intent = str(parsed.get("asking_about") or "").strip().lower()

    return Understanding(
        places=places,
        needs_city=needs_city,
        since=day("from"),
        until=day("to"),
        # FAILS CLOSED. An unknown or missing value refuses rather than
        # proceeds: a wrong refusal is visible to the user and they can
        # rephrase, a wrong crime answer is visible to nobody.
        asking_about=intent if intent in ASKING_ABOUT else "something_else",
        # `unclear` means NO PLACE AT ALL -- a dead end. An area without a city
        # is not that: it is a question we can answer once the user says which
        # city. Conflating them would turn "which city?" into "I don't
        # understand", and the user would have no way forward.
        unclear=(bool(parsed.get("unclear")) or not places) and not needs_city,
        cached=reply.cached,
    )
