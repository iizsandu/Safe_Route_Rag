"""The web app. Holds no data (D-031).

Loads three things once, at startup, and nothing per request:
    the embedding model   ~130 MB, needed to turn a question into a vector
    a Qdrant connection   the vectors and BM25 weights live there
    a DynamoDB connection the article text lives there (F-079)

That is what lets this run as several copies at once, which is the test
D-031 asks of every design -- with one accepted exception: the daily
question counter below is in-memory, per process, and does not honour
that test. See its own comment for why.

WHY THE ANSWER IS NOT STREAMED. The model returns structured incidents that
are then verified, repaired and rendered by us (D-018). The prose does not
exist until after verification, so there is nothing to stream. Streaming the
model's raw output would mean streaming text nobody has checked -- exactly
what D-018 exists to prevent.

Run:
    uvicorn rag.app:app --reload
    then open http://127.0.0.1:8000
"""

from __future__ import annotations

import html
import os
from contextlib import asynccontextmanager
from datetime import date as date_type
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from rag.articles import connect as connect_articles
from rag.embed import load_model
from rag.llm import ModelConfig
from rag.qdrant_search import connect
from rag.service import answer
from rag.understand import understand

MODEL = os.environ.get("RAG_MODEL", "nvidia/nemotron-3.5-lightning:free")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CACHE_DIR = Path("eval/responses")

DAILY_QUESTION_LIMIT = int(os.environ.get("DAILY_QUESTION_LIMIT", "15"))

# In-memory only -- resets on restart, and does not coordinate across
# multiple copies of this process (D-031's "many copies at once" test is
# knowingly not met here). Accepted for a single-instance deployment;
# revisit with a shared store if this ever runs as more than one process.
usage = {"date": None, "count": 0}

def over_daily_limit() -> bool:
    today = date_type.today().isoformat()
    if usage["date"] != today:
        usage["date"] = today
        usage["count"] = 0
    return usage["count"] >= DAILY_QUESTION_LIMIT

def record_question() -> None:
    usage["count"] += 1

state: dict = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load once, serve many. Per-request loading is what D-031 forbids."""
    load_model()                       # warm the embedding model
    state["client"] = connect(QDRANT_URL)
    state["articles"] = connect_articles()
    state["config"] = ModelConfig(model=MODEL)
    yield
    state.clear()

app = FastAPI(title="Safe Route", lifespan=lifespan)

PAGE = """<!doctype html>
<meta charset="utf-8"><title>Safe Route</title>
<style>
 body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 44rem;
        margin: 3rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.6rem; }}
 h2 {{ font-size: 1.2rem; margin: 2.4rem 0 .2rem; }}
 h3 {{ font-size: 1rem; font-weight: 600; margin: 0 0 .2rem; }}
 input {{ font: inherit; padding: .55rem .7rem; width: 26rem; }}
 button {{ font: inherit; padding: .55rem 1.1rem; }}
 article {{ border-top: 1px solid #e6e6e6; padding: 1rem 0; }}
 .meta {{ margin: 0; color: #555; font-size: .92rem; }}
 .src {{ margin: .35rem 0 0; font-size: .92rem; }}
 .src a {{ color: #1a4f9c; }}
 .sub {{ color: #666; font-size: .9rem; margin: 0; }}
 .dim {{ color: #888; border-bottom: 1px dotted #bbb; cursor: help; }}
 time {{ border-bottom: 1px dotted #bbb; cursor: help; }}
 .caveat {{ margin-top: 2rem; padding: .9rem 1rem; background: #f7f7f7;
            border-left: 3px solid #ccc; color: #444; font-size: .92rem; }}
 .none {{ font-size: 1rem; }}
 .stop {{ color: #a11; }}
 .summary {{ font-size: 1.08rem; margin: 1rem 0 1.2rem; }}
 details {{ border-top: 1px solid #e6e6e6; padding-top: .6rem; }}
 summary {{ cursor: pointer; color: #1a4f9c; font-size: .95rem; }}
 details article:first-of-type {{ border-top: none; }}
</style>
<h1>Safe Route</h1>
<form method="post" action="/ask">
  <input name="question" placeholder="Is it safe around Rohini, Delhi?"
         value="{question}" required autofocus>
  <button>Ask</button>
</form>
{body}
"""

# One block of copy, two refusals. They have different causes -- no place at
# all, versus a place but the wrong question -- but the reader needs the same
# thing from both: what this does, and how to ask it.
SCOPE = """<p><strong>{lead}</strong></p>
<p>This answers one kind of question: what crimes have been reported near a
place, according to news articles. It can&rsquo;t help with cafes, directions
or anything else.</p>
<p>Try: <em>&ldquo;Is it safe around Rohini, Delhi?&rdquo;</em></p>"""


def page(question: str = "", body: str = "") -> HTMLResponse:
    return HTMLResponse(PAGE.format(question=html.escape(question), body=body))

@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return page()

@app.get("/health")
def health() -> dict:
    """Alive AND able to reach Qdrant.

    A process that is running but cannot reach its database is not healthy,
    and reporting it as healthy is how an outage goes unnoticed -- the host
    keeps sending traffic to a container that can only fail.
    """
    try:
        count = state["client"].get_collection(
            "safe_route_chunks").points_count
        return {"ok": True, "chunks": count}
    except Exception as error:          # noqa: BLE001 -- any failure to reach
        return {"ok": False, "error": str(error)}   # Qdrant means unhealthy

@app.post("/ask", response_class=HTMLResponse)
def ask(question: str = Form(...)) -> HTMLResponse:
    """Understand, then search. Two model calls, and the first can stop us."""
    if over_daily_limit():
        return page(question, "<p><strong>This demo has reached today's "
                    "question limit.</strong> It runs on a small, shared, "
                    "free daily budget &mdash; please check back tomorrow.</p>")
    record_question()

    asked = understand(question, state["config"], CACHE_DIR)

    # Intent is checked FIRST, and it is the only place it can be checked.
    # "cafe near Rohini, Delhi" names a real place and passes every later
    # check -- by the time generate() runs the question is gone and only the
    # area remains, so no output check can ever see that it was off-topic.
    if asked.asking_about != "place_safety":
        lead = ("This can&rsquo;t answer questions about a named person."
                if asked.asking_about == "a_person"
                else "That isn&rsquo;t something this can answer.")
        return page(question, SCOPE.format(lead=lead))

    if asked.needs_city:
        # NEVER guess the city. "Rohini" alone retrieves Mumbai articles about
        # a woman of that name (F-066), and a confident answer about the wrong
        # place is the worst thing a safety product can do.
        areas = ", ".join(html.escape(a) for a in asked.needs_city)
        return page(question,
                    f"<p><strong>Which city is {areas} in?</strong> "
                    "Add it and ask again &mdash; there is more than one "
                    "place with that name, and guessing would give you a "
                    "confident answer about somewhere else.</p>")

    if not asked.ready:
        return page(question, SCOPE.format(
            lead="I could not find a place in that question."))

    # A route is several places. Each is searched on its own; nothing here
    # merges them yet.
    blocks = []
    for area in asked.places:
        blocks.append(one_area(area, asked))
    return page(question, "".join(blocks))

def one_area(area: str, asked) -> str:
    """The answer for a single place, as HTML."""
    result = answer(area, state["client"], state["articles"], state["config"], CACHE_DIR,
                    searched_from=asked.since or "2021-01-01",
                    searched_to=asked.until or "2026-08-29")

    if not result.sources:
        # "Nothing was indexed under this name" is NOT "this area is safe"
        # (D-011). The distinction is the product, so it is said in full.
        return (f"<h2>{html.escape(area)}</h2><p class='none'><strong>Nothing "
                "found.</strong> No article mentions it &mdash; that is not the "
                "same as the area being safe.</p>")

    if result.blocked:
        # What blocked it is logged, not shown. A reader cannot act on
        # "relative-phrase: article 87245822", and the honest thing to tell
        # them is that we are not confident enough to show it.
        for finding in result.findings:
            print(f"[blocked] {area}: {finding}", flush=True)
        return (f"<h2>{html.escape(area)}</h2><p class='stop'><strong>Answer "
                "withheld.</strong> The generated answer did not pass our "
                "checks, so we are not showing it.</p>")

    # Warnings go to the log too. They are for us; the answer is for them.
    for finding in result.findings:
        print(f"[warn] {area}: {finding}", flush=True)
    return result.html or ""
