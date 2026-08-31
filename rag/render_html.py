"""The same answer as rag/render.py, for a browser.

`render.py` writes for a terminal and for us: it labels every field and spells
out how each date was derived. That is the right output for debugging and the
wrong one for a reader.

Both are OURS, not the model's (D-018). The facts, the ordering and the
caveats are identical -- only the presentation differs. If they ever disagree
about what is safe to say, render.py is the reference.

Two things are deliberately NOT simplified away:
  * "date not stated" stays visible. Hiding it would let a reader assume we
    know when something happened (D-011).
  * the coverage caveat is printed in both branches, incidents or none. It is
    F-044 rendered as a sentence, not decoration.
"""

from __future__ import annotations

import html
from datetime import date
from typing import Any

from rag.dates import resolve
from rag.render import CAVEAT, STATUS_PHRASE, pretty

def _when(incident: dict[str, Any], published: date | None) -> str:
    """The date, with how we know it as a tooltip rather than a sentence.

    The explanation still exists -- hover the date -- because a reader who
    wants to know whether we resolved "saturday night" ourselves is entitled
    to find out. It just stops being the loudest thing on the line.
    """
    if published is None:
        return "<span class='dim'>date unknown</span>"

    stated = incident.get("incident_date")
    resolved, how = resolve(stated, incident.get("date_kind"), published)
    if resolved:
        return (f"<time datetime='{resolved.isoformat()}' "
                f"title='{html.escape(how)}'>{pretty(resolved)}</time>")

    if stated and incident.get("date_kind") != "not_stated":
        # The source DID say when; our parser could not read it. Say the
        # source's own words rather than blaming the source (F-074).
        return (f"<span class='dim' title='we could not resolve this to a "
                f"date'>{html.escape(str(stated))}</span>")
    return (f"<span class='dim' title='the article never says when this "
            f"happened; it was published {pretty(published)}'>"
            f"date not stated</span>")

RECENT_MONTHS = 12

def _summary(incidents: list[dict[str, Any]],
             published: dict[int, date], today: date) -> str:
    """The shape of the answer, counted rather than judged, with the recent
    claims themselves visible -- not just their count.

    Written HERE, not by the model (D-018, and the user's choice). Counting is
    something code does exactly and a model does approximately, and every time
    this model was asked to do one more thing the rest got worse (F-063,
    F-074). Listing the recent claims is the same principle applied one step
    further: the text is the model's own already-verified `claim`, joined to
    its date by code -- never a new sentence asked of the model. Asking the
    model to itself summarise "what happened recently" would put unverified
    prose in front of a reader (D-018's whole point), for something code can
    already do exactly from data verify.py has already checked.

    It says WHAT WAS REPORTED. It does not say whether the area is safe -- that
    would be a risk score built on newsroom coverage, which D-011 forbids and
    the caveat at the bottom exists to deny.

    Recency is separated because a 2021 shooting says less about walking there
    tomorrow than one from last month. Both are shown; neither is dropped.
    """
    cutoff = date(today.year - 1, today.month, min(today.day, 28))
    recent, older = [], []
    for incident in incidents:
        sources = [s for s in (incident.get("sources") or []) if isinstance(s, int)]
        when = published.get(sources[0]) if sources else None
        if when and when >= cutoff:
            recent.append((incident, when))
        else:
            older.append(when)

    if recent:
        # Most recent first -- what changed lately is what a reader most
        # wants to see without scrolling.
        ordered = sorted(recent, key=lambda pair: pair[1], reverse=True)
        items = "".join(
            f"<li>{_when(incident, when)} &mdash; "
            f"{html.escape(str(incident.get('claim') or '(no claim)'))}</li>"
            for incident, when in ordered)
        head = (f"<p class='summary'><strong>{len(recent)} incident"
                f"{'s' if len(recent) != 1 else ''} reported in the last "
                f"{RECENT_MONTHS} months:</strong></p>"
                f"<ul class='recent'>{items}</ul>")
    else:
        # The dangerous half of the question. Saying "nothing recent" invites
        # "so it is safe", and in this corpus that inference is unfounded --
        # roughly six real incidents a year for a MAJOR district (F-044), so
        # silence is the normal case. The sentence carries its own correction.
        head = (f"<p class='summary'><strong>No incidents reported in the "
                f"last {RECENT_MONTHS} months.</strong> That is not evidence "
                f"the area is safe &mdash; most areas report nothing in most "
                f"months.</p>")

    if older:
        years = sorted({w.year for w in older if w})
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}&ndash;{years[-1]}"
        head += (f"<p class='older'>{len(older)} older incident"
                 f"{'s' if len(older) != 1 else ''} ({span}).</p>")

    return head

def _incident(number: int, incident: dict[str, Any],
              published: dict[int, date], articles: dict[int, dict]) -> str:
    sources = [s for s in (incident.get("sources") or []) if isinstance(s, int)]
    first = published.get(sources[0]) if sources else None

    links = []
    for source in sources:
        article = articles.get(source, {})
        headline = html.escape(article.get("headline") or str(source))
        url = article.get("url") or ""
        links.append(f"<a href='{html.escape(url)}' target='_blank' "
                     f"rel='noopener'>{headline}</a>" if url else headline)

    return f"""<article>
  <h3>{number}. {html.escape(str(incident.get('claim') or '(no claim)'))}</h3>
  <p class="meta">{_when(incident, first)}
     &middot; {html.escape(str(incident.get('location') or 'location not specified'))}
     &middot; {html.escape(STATUS_PHRASE.get(incident.get('legal_status'),
                                             'legal status unclear'))}</p>
  <p class="src">{' &middot; '.join(links)}</p>
</article>"""

def render_html(
    incidents: list[dict[str, Any]],
    area: str,
    searched_from: date,
    searched_to: date,
    records: list[dict[str, Any]],
) -> str:
    """The complete answer as HTML. 'Nothing found' is a first-class result."""
    published = {r["article_id"]: date(*map(int, r["published_at"][:10].split("-")))
                 for r in records}
    articles = {r["article_id"]: r for r in records}
    window = f"{pretty(searched_from)} to {pretty(searched_to)}"
    place = html.escape(area)

    if not incidents:
        # NOT an all-clear. D-011's central rule, and the reason this branch
        # exists at all rather than rendering an empty list.
        thin = (f" {place} appears in only {len(records)} article"
                f"{'s' if len(records) != 1 else ''} we could find, so silence "
                f"here reflects thin coverage rather than safety."
                if len(records) <= 5 else "")
        return (f"<h2>{place}</h2>"
                f"<p class='summary'><strong>Nothing reported.</strong> No "
                f"crimes were reported near {place} in Times of India coverage "
                f"between {window}. This is not evidence the area is safe "
                f"&mdash; only that nothing was published.{thin}</p>"
                f"<p class='caveat'>{html.escape(CAVEAT)}</p>")

    body = "".join(_incident(n, i, published, articles)
                   for n, i in enumerate(incidents, start=1))
    # The incidents go in a fold. The summary is what a reader needs; the
    # sources are what they need to CHECK it, and both must be present --
    # a claim nobody can verify is not evidence (CLAUDE.md section 9).
    return (f"<h2>{place}</h2>"
            f"<p class='sub'>Times of India, {window} &middot; "
            f"{len(records)} articles examined</p>"
            f"{_summary(incidents, published, searched_to)}"
            f"<details><summary>Sources ({len(incidents)})</summary>{body}</details>"
            f"<p class='caveat'>{html.escape(CAVEAT)}</p>")
