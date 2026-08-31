"""Phase 11 item 1 -- does a claim assert guilt while legal_status is unproven?

Built after this failure was confirmed live (F-078, article 80260013): the
source hedged with "accused"; the model's claim read "Vinod Trivedi assaulted
and murdered Sadhana Chowdhary", flat fact, and nothing existing caught it.

This also documents a KNOWN false positive on purpose (case 5) rather than
hiding it: Indian crime headlines routinely compress the passive voice
without "was" -- "Rajesh Kumar shot dead" -- where the named person is the
VICTIM, in the exact position this heuristic expects a perpetrator. That is
why check_guilt_language is WARNING, not ERROR (verify.py's own docstring
reserves ERROR for what is free and certain; this is neither).

Run:
    python tests/test_verify_guilt.py
"""

from __future__ import annotations

from rag.generate import Answer
from rag.verify import WARNING, check_guilt_language

CASES = [
    ("real failure case (F-078)",
     "Carpenter Vinod Trivedi assaulted and murdered Sadhana Chowdhary, "
     "director of a cooperative credit society, with a hammer.",
     "reported", [WARNING]),

    ("correctly hedged",
     "Two men were arrested for allegedly bludgeoning to death a "
     "30-year-old man over suspicion of him having an extramarital affair.",
     "reported", []),

    ("victim only, no name attached",
     "A woman was murdered near Rohini.",
     "reported", []),

    ("guilt established -- conviction exempts it",
     "Vinod Trivedi murdered Sadhana Chowdhary with a hammer.",
     "convicted", []),

    ("KNOWN false positive -- victim named, compressed passive",
     "Rajesh Kumar shot dead in an alleged police encounter.",
     "reported", [WARNING]),
]


def run() -> None:
    failures = []
    for label, claim, legal_status, expected_severities in CASES:
        answer = Answer(
            incidents=[{"claim": claim, "legal_status": legal_status}],
            sent_ids=[1], raw_text="", cached=False)
        findings = check_guilt_language(answer)
        got = sorted(f.severity for f in findings)
        want = sorted(expected_severities)
        if got != want:
            failures.append(
                f"{label}: expected {want}, got {got} "
                f"({[str(f) for f in findings]})")

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        raise SystemExit(f"\n{len(failures)} of {len(CASES)} checks failed")

    print(f"PASS  all {len(CASES)} guilt-language checks behaved as expected "
          f"(including the documented false positive)")


if __name__ == "__main__":
    run()
