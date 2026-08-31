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
    # Which incident this is about. Set only by checks whose damage can be
    # undone by removing a field -- repair() needs to know WHICH incident.
    article_id: int | None = None

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
                    f"its publication date but labelled 'explicit'",
                    source))
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
                    f"the renderer would present it as stated by the source.",
                    source))
    return findings


# Words that assert a legal step has happened. Matched against the model's own
# `claim` text, never against the source -- the question is whether the model
# contradicts ITSELF, which needs no reading judgement (F-074).
STATUS_WORDS = {
    "arrested": {"arrested", "charged", "convicted", "acquitted"},
    "convicted": {"convicted"},
    "acquitted": {"acquitted"},
    "charged": {"charged", "convicted", "acquitted"},
}

WEEKDAY_IN_CLAIM = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"yesterday|last night|this morning)\b", re.IGNORECASE)

# Indian mobile numbers: 10 digits, starting 6-9, optionally prefixed +91.
# Lookaround excludes it matching PART of a longer number (an article id,
# a date run together) rather than a real 10-digit phone number.
PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
VIDEO_WORDS = re.compile(r"\b(video|photo|clip|footage|picture)\b", re.IGNORECASE)

# A capitalised name (1-3 words) directly followed by a guilt verb, with
# nothing between them -- a hedge word like "allegedly" or "accused of"
# breaks the match on purpose, since it sits between the name and the verb.
GUILT_VERBS = ("murdered", "killed", "raped", "assaulted", "robbed",
              "molested", "strangled", "stabbed", "shot", "kidnapped")
NAME_THEN_GUILT_VERB = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(" +
    "|".join(GUILT_VERBS) + r")\b")


def check_status_agrees_with_claim(answer: Any) -> list[Finding]:
    """The claim says an arrest happened; `legal_status` says it did not.

    F-074: two incidents rendered as "65 people were ARRESTED..." directly above
    "reported; no arrest or charge stated". Both lines came from the same model
    in the same reply, and a reader sees them together.

    This compares two things the MODEL wrote, so it is free and certain -- unlike
    D-018 failure mode 2, which needs someone to read the source. `unknown` is
    exempt: it is an honest admission, not a contradiction.

    WARNING, not ERROR (decided 2026-08-29). It shipped as an ERROR for exactly
    one run, which withheld nine correct incidents because two carried a
    mislabelled status. Two reasons that severity was wrong:

      * the mistake runs in the CAUTIOUS direction. The status UNDERSTATES what
        happened -- "no arrest or charge stated" printed under a claim saying
        arrested -- so it never asserts guilt or an arrest that did not occur.
        CLAUDE.md section 9 is concerned with the opposite direction.
      * blocking is not free. Showing nothing to someone asking whether an area
        is safe has its own cost, and D-011 is explicit that silence must never
        be read as reassurance.

    Raise it back to ERROR only if a case appears where the claim OVERSTATES the
    legal step -- that direction does justify withholding the answer.
    """
    findings = []
    for incident in answer.incidents:
        claim = str(incident.get("claim") or "").lower()
        status = str(incident.get("legal_status") or "")
        for word, permitted in STATUS_WORDS.items():
            if word in claim and status not in permitted | {"unknown"}:
                findings.append(Finding(
                    "status-agrees", WARNING,
                    f"claim says {word!r} but legal_status is {status!r}: "
                    f"{str(incident.get('claim'))[:80]!r}"))
                break
    return findings


def check_date_agrees_with_claim(answer: Any) -> list[Finding]:
    """The claim names a day; `date_kind` says no date was stated.

    F-074: three incidents whose claims read "on Saturday night", "on Friday and
    Saturday", "on Sunday" were all marked `not_stated`, so the renderer printed
    "date not stated" beside a claim naming a day.

    A WARNING, not an error. The model may have written a weekday into the claim
    that the source attaches to something other than the incident -- an arrest,
    a hearing. Worth surfacing; not worth blocking an answer over.
    """
    findings = []
    for incident in answer.incidents:
        claim = str(incident.get("claim") or "")
        if incident.get("date_kind") != "not_stated":
            continue
        found = WEEKDAY_IN_CLAIM.search(claim)
        if found:
            findings.append(Finding(
                "date-agrees", WARNING,
                f"claim says {found.group(0)!r} but date_kind is 'not_stated', "
                f"so the answer will read 'date not stated': {claim[:70]!r}"))
    return findings


# Findings whose damage is confined to one field of one incident, and can be
# undone by removing that field. Everything else stays fatal.
#
# "date-kind" is here even though it is only a WARNING. Article 90421071 said
# "On Wednesday" and was published on a Thursday; the model returned the
# PUBLICATION date labelled 'explicit', and the answer told the reader the
# source had stated it. Off by a day, and falsely attributed. That is exactly
# D-011's trap, so the date is dropped rather than shown.
REPAIRABLE = {"relative-phrase", "explicit-date", "date-kind"}


def repair(answer: Any, findings: list[Finding]) -> list[Finding]:
    """Strip fabricated dates instead of withholding the whole answer.

    One invented date used to block every incident in the reply -- four good
    incidents hidden because a fifth carried a bad date (F-074). The reader
    then saw nothing, and D-011 is explicit that silence must never be read as
    reassurance.

    The fabrication does not reach the reader either way. What changes is
    whether the rest of the answer does.

    Repair is only safe where removing the field leaves a still-true incident.
    A date can go; a citation cannot -- an uncited claim is not a weaker claim,
    it is an ungrounded one, so those stay fatal.
    """
    repaired = []
    for finding in findings:
        if finding.check not in REPAIRABLE or finding.article_id is None:
            repaired.append(finding)
            continue
        for incident in answer.incidents:
            if finding.article_id in (incident.get("sources") or []):
                incident["incident_date"] = None
                incident["date_kind"] = "not_stated"
        repaired.append(Finding(
            finding.check, WARNING,
            f"REPAIRED -- date removed from the answer. {finding.detail}",
            finding.article_id))
    return repaired


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
                + check_explicit_dates(answer, texts)
                # F-074: the first two checks that compare a structured field
                # against the model's own claim text, rather than checking the
                # field in isolation. F-056's blind spot, narrowed slightly.
                + check_status_agrees_with_claim(answer)
                + check_date_agrees_with_claim(answer)
                + check_privacy(answer)
                + check_guilt_language(answer))

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

def check_privacy(answer: Any) -> list[Finding]:
    """CLAUDE.md section 9: a source naming someone does not mean our
    sentence should repeat it. A phone number is certain and blocked. A
    video/photo mention is a judgement call -- logged for a human, not
    blocked, since a claim can legitimately mention evidence without
    pointing at circulated material.

    What this does NOT do: catch a PROTECTED NAME -- a sexual-offence
    victim, a minor, or a witness (D-034 narrowed this from "any victim"
    once "never name a victim" was found suppressing an ordinary, legally
    named murder victim). Catching it needs knowing WHICH name in the
    sentence is protected, which is the entity-resolution problem F-047
    already found this corpus does not support. Rule 7 of the generation
    prompt is the only defence against that; its failure would not be
    caught here, and that gap is accepted, not hidden.
    """
    findings = []
    for incident in answer.incidents:
        claim = str(incident.get("claim") or "")
        if PHONE.search(claim):
            findings.append(Finding(
                "privacy-phone", ERROR,
                f"claim contains what looks like a phone number: {claim[:80]!r}"))
        if VIDEO_WORDS.search(claim):
            findings.append(Finding(
                "privacy-video-ref", WARNING,
                "claim references video/photo/footage -- confirm it does not "
                f"point a reader at circulated material: {claim[:80]!r}"))
    return findings

def check_guilt_language(answer: Any) -> list[Finding]:
    """CLAUDE.md section 9, first line: "never transform 'police accused X
    of...' into 'X committed...'". Confirmed happening live (F-078, article
    80260013): the source hedged with "accused"; `claim` read "Vinod Trivedi
    assaulted and murdered Sadhana Chowdhary", flat fact, and nothing else in
    this file caught it -- check_status_agrees_with_claim only looks for the
    words arrested/charged/convicted/acquitted IN THE CLAIM, and none of
    those appear when the claim uses a guilt verb instead.

    WARNING, not ERROR, by design (2026-08-30) -- this is a heuristic, not a
    certainty, and it has a known, likely-common false positive in THIS
    corpus: Indian crime headlines routinely compress the passive voice
    without "was" -- "Rajesh Kumar shot dead", "Woman stabbed to death" --
    where the named person is the VICTIM, sitting in exactly the position
    this pattern expects a perpetrator. verify.py's own rule is that ERROR
    is for what is free AND CERTAIN (see module docstring); this is neither,
    at n=1 confirmed case. Revisit toward ERROR once several real cases
    confirm the false-positive rate is low enough to justify blocking on it.

    Known blind spot, stated rather than hidden: this cannot see guilt
    asserted in the other sentence order -- "the murder of X by Y" -- only
    "Y murdered X".
    """
    findings = []
    for incident in answer.incidents:
        if str(incident.get("legal_status") or "") == "convicted":
            continue
        claim = str(incident.get("claim") or "")
        match = NAME_THEN_GUILT_VERB.search(claim)
        if match:
            findings.append(Finding(
                "guilt-language", WARNING,
                f"{match.group(1)!r} directly precedes the guilt verb "
                f"{match.group(2)!r} while legal_status is "
                f"{incident.get('legal_status')!r}, not convicted: "
                f"{claim[:80]!r}"))
    return findings

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
                    f"resolve it to a specific day.",
                    source))
    return findings
