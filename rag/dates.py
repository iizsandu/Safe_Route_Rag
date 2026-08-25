"""Resolve the date phrases the model returns into real dates.

The model is told never to convert "on Friday" into a calendar date, because
it does the arithmetic badly and labels the guess as a stated fact (F-054).
It copies the phrase; this module does the arithmetic, where it is
deterministic and unit-testable without an API call.

Scope is set by what the model ACTUALLY produces (F-057): every relative phrase
across six runs was a weekday name. Offset phrases ("two weeks ago") appear in
108 of 790 corpus articles but have never yet been emitted, so they are not
handled -- and return None rather than a guess.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

def resolve_weekday(phrase: str, published: date) -> date | None:
    """"friday" -> the most recent Friday on or before the publication date.

    News reports past events, so "on Friday" in an article printed on Sunday
    means two days ago, never the coming Friday. If the article is printed on
    the same weekday it names, that is the same day -- article 89719546 was
    printed on Monday 2022-02-21 and says "on Monday", meaning that morning.
    """
    target = WEEKDAYS.get(phrase.strip().lower())
    if target is None:
        return None
    days_back = (published.weekday() - target) % 7
    return published - timedelta(days=days_back)

def resolve_month_day(phrase: str, published: date) -> date | None:
    """"february 22" -> a real date, taking the year from the publication date.

    If that lands after publication, the article must mean the previous year --
    a story printed in March cannot describe an incident the following December.
    """
    match = re.match(r"([a-z]+)\s+(\d{1,2})$", phrase.strip().lower())
    if not match:
        return None
    month = MONTHS.get(match.group(1))
    if month is None:
        return None
    try:
        candidate = date(published.year, month, int(match.group(2)))
    except ValueError:
        return None
    return candidate if candidate <= published else candidate.replace(
        year=published.year - 1)

def resolve(
    incident_date: str | None,
    date_kind: str | None,
    published: date,
) -> tuple[date | None, str]:
    """Return (resolved date or None, how it was derived).

    The second value exists so an answer can say HOW it knows a date, and so an
    unresolvable phrase is visible rather than silently becoming None. Never
    falls back to the publication date -- that is the D-011 trap the whole
    module exists to avoid.
    """
    if not incident_date or date_kind == "not_stated":
        return None, "not stated in the article"

    text = str(incident_date).strip().lower()

    iso = re.match(r"(\d{4})-(\d{2})-(\d{2})$", text)
    if iso:
        try:
            return date(*map(int, iso.groups())), "stated in the article"
        except ValueError:
            return None, f"unparseable date {incident_date!r}"

    resolved = resolve_month_day(text, published)
    if resolved:
        return resolved, "stated in the article, year from publication date"

    resolved = resolve_weekday(text, published)
    if resolved:
        return resolved, f"{text} before publication on {published}"

    return None, f"unrecognised phrase {incident_date!r}"
