"""Turn verified incidents into the text a person reads.

The MODEL never writes this text -- we do (D-018). That is what makes
"nothing reported is not the same as safe" enforceable in code rather than
dependent on a model choosing good words, which it has failed to do in six
different ways today.

Two rules are mechanical here:
  1. every date says WHERE IT CAME FROM. A date we could not establish is
     said to be unknown, never quietly replaced by the publication date
     (D-011).
  2. the coverage caveat is printed in BOTH branches -- incidents found and
     nothing found. It is not decoration; it is F-044 rendered as a sentence.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from rag.dates import resolve

# Rendered as a separate labelled line, never woven into the claim sentence.
# Rewriting the model's wording risks changing its meaning; stating the status
# beside it does not (CLAUDE.md section 9: allegation is not guilt).
STATUS_PHRASE = {
    # "reported" means the source states the incident and nothing more. It must
    # not imply a complainant: 3 of 10 incidents in the Rohini answer are police
    # encounters, where nobody reported anything to police.
    "reported": "reported; no arrest or charge stated",
    "accused": "someone has been ACCUSED -- not convicted",
    "arrested": "an arrest was made -- not a conviction",
    "charged": "charges filed -- not a conviction",
    "convicted": "a conviction was recorded",
    "acquitted": "the accused was ACQUITTED",
    "unknown": "legal status not stated in the source",
}

CAVEAT = (
    "This reflects news reporting, not police records. Newspapers report "
    "murder and assault, not snatching or harassment, so nothing here is "
    "evidence that an area is safe."
)


def pretty(day: date) -> str:
    """25 Jun 2025."""
    return f"{day.day} {day.strftime('%b %Y')}"

def format_when(incident: dict[str, Any], published: date) -> str:
    """Say when the incident happened, and how we know.

    Option (a) from the design discussion: an incident whose date is not
    stated is still REPORTED, with the gap made explicit. Omitting it would
    hide a real crime; hedging from the publication date would be the exact
    trap D-011 warns about, wearing a hedge.

    THREE states, not two (F-074). Collapsing the middle one into "not stated"
    made this renderer assert something false about a source: article 84510053
    said "Friday night", `dates.py` could not resolve it because WEEKDAYS is an
    exact lookup, and the reader was told the date was not stated. Source
    provenance is a hard requirement (CLAUDE.md section 9), and the component
    that exists to protect it was the one breaking it.
    """
    stated = incident.get("incident_date")
    resolved, how = resolve(stated, incident.get("date_kind"), published)

    if resolved:
        return f"{pretty(resolved)}  ({how})"
    if stated and incident.get("date_kind") != "not_stated":
        # The source DID say when. We failed to turn it into a date, and say so
        # rather than blaming the source for our parser.
        return (f"source says {str(stated)!r}, which we could not resolve to a "
                f"date; article published {pretty(published)}")
    return f"date not stated; article published {pretty(published)}"

def render_incident(
    number: int,
    incident: dict[str, Any],
    published: dict[int, date],
    headlines: dict[int, str],
) -> str:
    """One incident as four labelled lines plus its sources."""
    sources = [s for s in (incident.get("sources") or []) if isinstance(s, int)]
    first = published.get(sources[0]) if sources else None

    lines = [f"{number}. {incident.get('claim', '(no claim)')}"]
    lines.append(f"   when    {format_when(incident, first) if first else 'unknown'}")
    lines.append(f"   where   {incident.get('location') or 'not specified'}")
    lines.append(f"   status  {STATUS_PHRASE.get(incident.get('legal_status'), 'unrecognised status')}")
    for source in sources:
        lines.append(f"   source  [{source}] {headlines.get(source, '')}")
    return "\n".join(lines)

def render(
    incidents: list[dict[str, Any]],
    area: str,
    searched_from: date,
    searched_to: date,
    records: list[dict[str, Any]],
) -> str:
    """The complete answer. Handles 'nothing found' as a first-class result."""
    published = {r["article_id"]: date(*map(int, r["published_at"][:10].split("-")))
                 for r in records}
    headlines = {r["article_id"]: r.get("headline") or "" for r in records}
    window = f"{pretty(searched_from)} to {pretty(searched_to)}"
    span = f"{pretty(searched_from)} and {pretty(searched_to)}"

    if not incidents:
        return "\n".join([
            f"No crimes were reported near {area} in Times of India coverage "
            f"between {span}.",
            "",
            f'This is NOT the same as "{area} is safe". It means nothing '
            f"appeared in this one newspaper's reporting. Most areas, in most "
            f"months, return nothing.",
            "",
            CAVEAT,
        ])

    header = [
        f"Crimes reported near {area}",
        f"Times of India coverage, {window} · {len(records)} articles examined",
        "",
    ]
    body = [render_incident(n, i, published, headlines)
            for n, i in enumerate(incidents, start=1)]
    return "\n".join(header + ["\n".join(body), "", CAVEAT])
