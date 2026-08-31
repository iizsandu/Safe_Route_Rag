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
from fastapi.staticfiles import StaticFiles

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
app.mount("/static", StaticFiles(directory="rag/static"), name="static")

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safe Route</title>
  <style>
    :root {{
      --ink: #f0e7d5;
      --muted: #aaa18f;
      --navy: #07121b;
      --amber: #d5a45b;
      --line: rgba(213, 164, 91, 0.55);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--navy);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}

    main {{
      position: relative;
      display: grid;
      min-height: 100vh;
      place-items: center;
      overflow: hidden;
      padding: 2rem;
      isolation: isolate;
    }}

    main::before {{
      position: absolute;
      z-index: -2;
      inset: 0;
      background: #07121b;
      content: "";
    }}

    main::after {{
      position: absolute;
      z-index: -1;
      inset: 0;
      background-image: url("/static/india-archive-bg.png");
      background-position: center;
      background-repeat: no-repeat;
      background-size: cover;
      content: "";
      opacity: 0.14;
      pointer-events: none;
    }}

    .hero {{
      width: min(100%, 760px);
      text-align: center;
    }}

    .wordmark {{
      margin: 0 0 1.5rem;
      color: var(--ink);
      font-size: 1.55rem;
      font-weight: 400;
      letter-spacing: -0.03em;
    }}

    .wordmark::after {{
      display: block;
      width: 2rem;
      height: 1px;
      margin: 0.9rem auto 0;
      background: var(--amber);
      content: "";
    }}

    h1 {{
      max-width: 700px;
      margin: 0 auto;
      font-size: clamp(2.8rem, 7vw, 5.4rem);
      font-weight: 400;
      letter-spacing: -0.05em;
      line-height: 0.98;
    }}

    .intro {{
      margin: 1.25rem 0 2.25rem;
      color: var(--muted);
      font-size: clamp(1rem, 2vw, 1.25rem);
    }}

    form {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      width: min(100%, 600px);
      margin: 0 auto;
    }}

    input {{
      min-width: 0;
      padding: 1rem 1.2rem;
      border: 1px solid var(--line);
      border-right: 0;
      border-radius: 0;
      background: rgba(7, 18, 27, 0.82);
      color: var(--ink);
      font: inherit;
      font-size: 1rem;
    }}

    input::placeholder {{
      color: #9e957f;
    }}

    input:focus {{
      outline: 2px solid var(--amber);
      outline-offset: 3px;
    }}

    button {{
      padding: 1rem 1.6rem;
      border: 1px solid var(--amber);
      border-radius: 0;
      background: var(--amber);
      color: #15110b;
      cursor: pointer;
      font: inherit;
      font-size: 1rem;
      font-weight: 700;
    }}

    button:hover {{
      background: #e3b86f;
    }}

    .results {{
      width: min(100%, 760px);
      margin-top: 3rem;
      color: var(--ink);
      text-align: left;
    }}

    .results h2 {{
      margin-top: 3rem;
      font-size: 1.7rem;
      font-weight: 400;
    }}

    .results article {{
      padding: 1.2rem 0;
      border-top: 1px solid rgba(240, 231, 213, 0.18);
    }}

    .results a {{
      color: var(--amber);
    }}

    .results .caveat {{
      margin-top: 2rem;
      padding: 1rem;
      border-left: 2px solid var(--amber);
      color: var(--muted);
    }}

    .results .stop {{
      color: #e2b3a8;
    }}

    .tutorial {{
      max-width: 640px;
      margin: 0 auto 2rem;
      color: var(--muted);
      font-size: 0.9rem;
      text-align: center;
    }}

    .tutorial p {{
      margin: 0 0 0.6rem;
      line-height: 1.7;
    }}

    .chip {{
      padding: 0;
      border: none;
      background: none;
      color: var(--amber);
      font: inherit;
      font-size: inherit;
      text-decoration: underline;
      text-underline-offset: 2px;
      cursor: pointer;
    }}

    .chip:hover {{
      color: #e3b86f;
    }}

    .tutorial-hide {{
      display: inline-block;
      margin-top: 0.5rem;
      padding: 0.35rem 0.9rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-size: 0.8rem;
      cursor: pointer;
    }}

    .tutorial-hide:hover {{
      border-color: var(--amber);
      color: var(--amber);
    }}

    @media (max-width: 560px) {{
      main {{
        padding: 1.5rem;
      }}

      form {{
        grid-template-columns: 1fr;
        gap: 0.75rem;
      }}

      input {{
        border-right: 1px solid var(--line);
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="wordmark">Safe Route</p>

      <h1>A small step in making India Crime Aware.</h1>

      <p class="intro">
        Which place would you like to know more about?
      </p>

      <div id="tutorial-box" class="tutorial" style="display:none;">
        <p>New here? Try
          <button type="button" class="chip"
                  data-question="Is it safe around Rohini, Delhi?">Rohini, Delhi</button>,
          <button type="button" class="chip"
                  data-question="Is it safe around Sultanpur, Delhi?">Sultanpur, Delhi</button>,
          or
          <button type="button" class="chip"
                  data-question="Rohini, Delhi to Saket, Delhi">a route</button>.
        </p>
        <button type="button" id="tutorial-close"
                class="tutorial-hide">Hide this</button>
      </div>
      <script>
        (function () {{
          var box = document.getElementById('tutorial-box');
          if (!box) return;
          if (!localStorage.getItem('saferoute_tutorial_seen')) {{
            box.style.display = 'block';
          }}
          var input = document.getElementById('question');
          var chips = box.querySelectorAll('.chip');
          for (var i = 0; i < chips.length; i++) {{
            chips[i].addEventListener('click', function (event) {{
              if (input) {{
                input.value = event.target.getAttribute('data-question');
                input.focus();
              }}
            }});
          }}
          var hide = document.getElementById('tutorial-close');
          if (hide) {{
            hide.addEventListener('click', function () {{
              box.style.display = 'none';
              try {{
                localStorage.setItem('saferoute_tutorial_seen', '1');
              }} catch (e) {{}}
            }});
          }}
        }})();
      </script>

      <form method="post" action="/ask">
        <input
          id="question"
          name="question"
          type="text"
          value="{question}"
          placeholder="Search an area or route"
          required
          autofocus
        >
        <button type="submit">Search</button>
      </form>
    </section>

    <section class="results" aria-live="polite">
      <!-- RESULTS INSERT HERE -->
      {body}
    </section>
  </main>
</body>
</html>
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
