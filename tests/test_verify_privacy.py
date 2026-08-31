"""Phase 11 item 2 -- the two checks in verify.check_privacy() that ARE
closed and certain: a phone number (blocks) and a video/photo mention
(logs, does not block).

Does NOT test whether a victim's name is caught, because nothing does.
check_privacy()'s own docstring says so: that needs knowing which name in
the sentence is the victim's, which is the entity-resolution problem F-047
already found this corpus cannot support. That gap is a prompt rule only
(generate.py rule 7) and is not exercised here.

Run:
    python tests/test_verify_privacy.py
"""

from __future__ import annotations

from rag.generate import Answer
from rag.verify import ERROR, WARNING, check_privacy

CASES = [
    ("clean", "A man was assaulted near Rohini market.", []),
    ("phone", "Call 9876543210 about the assault near Rohini.", [ERROR]),
    ("video", "A video of the assault near Rohini was shared online.", [WARNING]),
]


def run() -> None:
    failures = []
    for label, claim, expected_severities in CASES:
        answer = Answer(incidents=[{"claim": claim}], sent_ids=[1],
                        raw_text="", cached=False)
        findings = check_privacy(answer)
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

    print(f"PASS  all {len(CASES)} privacy checks behaved as expected")


if __name__ == "__main__":
    run()
