"""Check a generated answer against what was actually sent to the model.

Nothing here calls an API. These are the checks that are FREE and CERTAIN
(D-018): a fabricated citation, an uncited claim, a source that vanished. The
one check that cannot be done in code -- does the cited article actually
SUPPORT the claim -- is deliberately absent, and its absence is stated rather
than hidden.

A failed verification is itself an abstention: an answer that does not pass
must not be shown to a user.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Rule 4 of the prompt constrains these. A field that accepts arbitrary values
# cannot be filtered on downstream, so drift is worth catching early.
LEGAL_STATUS = {"reported", "accused", "arrested", "charged",
                "convicted", "acquitted", "unknown"}
DATE_KIND = {"explicit", "relative", "not_stated"}
ERROR = "error"
WARNING = "warning"


@dataclass
class Finding:
    """One problem with an answer."""

    check: str
    severity: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():7}] {self.check}: {self.detail}"
    
def check_citations_exist(answer: Any) -> list[Finding]:
    """Every cited article_id must be one we actually sent.

    This is D-018's failure mode 1, and the only citation check that is
    certain: the set of ids sent is known exactly, so anything outside it is
    fabricated. Cannot produce a false alarm.
    """
    sent = set(answer.sent_ids)
    findings = []

    for incident in answer.incidents:
        for source in incident.get("sources") or []:
            if source not in sent:
                findings.append(Finding(
                    "citation-exists", ERROR,
                    f"incident cites article_id {source}, which was never sent. "
                    f"Sent: {sorted(sent)}"))

    return findings

def check_claims_cited(answer: Any) -> list[Finding]:
    """Every incident must carry at least one source. D-018 failure mode 3."""
    return [
        Finding("claim-cited", ERROR,
                f"incident has no sources: {incident.get('claim', '')[:80]!r}")
        for incident in answer.incidents
        if not (incident.get("sources") or [])
    ]

def check_reconciliation(answer: Any) -> list[Finding]:
    """Report sources that produced no incident.

    This is INFORMATION, not an error. An earlier schema demanded a strict
    partition -- every source either an incident or explicitly excluded -- and
    the model kept putting articles in both lists. It was right to: an article
    covering nine shootouts in five places is genuinely part in scope and part
    out. The partition was our invention, and removing it removed the error.

    A source producing no incident is now indistinguishable from one the model
    skipped. That is a real loss of information, accepted knowingly: a wrong
    answer we cannot detect is worse than a gap we can see.
    """
    if not answer.unused_ids:
        return []
    return [Finding(
        "reconciliation", WARNING,
        f"{len(answer.unused_ids)} of {len(answer.sent_ids)} sources produced "
        f"no incident: {answer.unused_ids}. Not necessarily wrong -- but we "
        f"cannot tell 'read and rejected' from 'skipped'.")]


def check_vocabulary(answer: Any) -> list[Finding]:
    """Enumerated fields must use the values the prompt allowed."""
    checks = [("legal_status", LEGAL_STATUS),
              ("date_kind", DATE_KIND)]
    findings = []

    for incident in answer.incidents:
        for field, allowed in checks:
            value = incident.get(field)
            if value is not None and value not in allowed:
                findings.append(Finding(
                    "vocabulary", WARNING,
                    f"{field}={value!r} is not in {sorted(allowed)}"))

    return findings

def check_date_kinds(answer: Any, published: dict[int, str]) -> list[Finding]:
    """Flag an incident date that is 'explicit' but equals the PRINT date.

    Rule 3 forbids inferring the incident date from `published`. When the two
    match exactly and the model called it explicit, the likeliest explanation is
    that it did infer it -- and labelled a guess as a stated fact, which is worse
    than admitting the date is unknown.

    A WARNING, not an error: an article can legitimately state a date that
    happens to be its publication date.
    """
    findings = []
    for incident in answer.incidents:
        if incident.get("date_kind") != "explicit":
            continue
        claimed = incident.get("incident_date")
        for source in incident.get("sources") or []:
            if claimed and claimed == published.get(source):
                findings.append(Finding(
                    "date-kind", WARNING,
                    f"article {source}: incident_date {claimed} is identical to "
                    f"its publication date but labelled 'explicit'"))
    return findings

DATE_EVIDENCE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"yesterday|today|tonight|\d{4})\b", re.I)


def check_explicit_dates(answer: Any, texts: dict[int, str]) -> list[Finding]:
    """An "explicit" date requires the article to contain SOME date reference.

    date_kind "explicit" claims the article prints a date. If the source has no
    month name, no weekday and no year anywhere, that claim cannot be true --
    and rag/render.py prints it as "(stated in the article)", attributing a
    fabrication to a real source with a real citation beside it.

    Article 122075852 says only "Two weeks after armed men fired...". The model
    returned 2025-06-25 -- its own publication date -- labelled "explicit".

    Deliberately conservative: fires only when the article has NO date evidence
    at all, so an article that does mention dates never false-alarms here. The
    weaker "explicit date equals the publication date" case stays a WARNING in
    check_date_kinds.
    """
    findings = []
    for incident in answer.incidents:
        if incident.get("date_kind") != "explicit":
            continue
        for source in incident.get("sources") or []:
            text = texts.get(source, "")
            if text and not DATE_EVIDENCE.search(text):
                findings.append(Finding(
                    "explicit-date", ERROR,
                    f"article {source}: date_kind 'explicit' but the source "
                    f"contains no date reference at all -- no month, weekday or "
                    f"year. {incident.get('incident_date')!r} is fabricated, and "
                    f"the renderer would present it as stated by the source."))
    return findings


def verify(answer: Any, records: list[dict[str, Any]]) -> list[Finding]:
    """Run every free check. Returns findings, most severe first."""
    published = {r["article_id"]: (r.get("published_at") or "")[:10]
                 for r in records}

    texts = {r["article_id"]: ((r.get("headline") or "") + " " +
                               (r.get("body_text") or "")) for r in records}
    
    findings = (check_citations_exist(answer)
                + check_claims_cited(answer)
                + check_reconciliation(answer)
                + check_vocabulary(answer)
                + check_date_kinds(answer, published)
                + check_relative_phrases(answer, texts)
                + check_explicit_dates(answer, texts))

    return sorted(findings, key=lambda f: f.severity != ERROR)


def report(findings: list[Finding]) -> bool:
    """Print findings. Returns True if the answer is safe to show."""
    errors = [f for f in findings if f.severity == ERROR]
    warnings = [f for f in findings if f.severity == WARNING]

    for finding in findings:
        print(f"  {finding}")

    print(f"\n  {len(errors)} error(s), {len(warnings)} warning(s)")
    print("\n  NOT CHECKED: whether each cited article actually SUPPORTS its")
    print("  claim (D-018 failure mode 2). That needs a reading judgement --")
    print("  an LLM judge or a human. Its absence is a known gap, not a pass.")

    return not errors

def check_relative_phrases(answer: Any, texts: dict[int, str]) -> list[Finding]:
    """A "relative" date phrase must actually appear in its source article.

    date_kind "relative" means the phrase was COPIED from the article. If it is
    not there, the model invented it -- and rag/dates.py will faithfully convert
    the invention into a precise date, which reads as more reliable than the
    guess it came from. Article 122075852 has no weekday word anywhere in its
    body; the model returned "friday", which resolved to 2025-06-20.

    Free and certain: a substring search over text we already hold.
    """
    findings = []
    for incident in answer.incidents:
        if incident.get("date_kind") != "relative":
            continue
        phrase = str(incident.get("incident_date") or "").strip().lower()
        if not phrase:
            continue
        for source in incident.get("sources") or []:
            text = texts.get(source, "").lower()
            if text and phrase not in text:
                findings.append(Finding(
                    "relative-phrase", ERROR,
                    f"article {source}: date phrase {phrase!r} does not appear "
                    f"in the source. The model invented it, and dates.py will "
                    f"resolve it to a specific day."))
    return findings
