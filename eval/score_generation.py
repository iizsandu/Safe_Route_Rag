"""Score the GENERATOR against the human answer key.

rag/verify.py asks "is what is here valid?". Nothing asked "is anything
missing?" -- so an empty incidents list passes every check (F-056). This is the
missing half: RECALL, measured against eval/rohini_labels.csv.

Reads saved responses only. No API calls.

Run:
    python eval/score_generation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag.generate import extract_json, resolve_labels

NORM = {"relevant": "relevant", "not relevant": "not_relevant",
        "not_relevant": "not_relevant", "irrelevant": "not_relevant",
        "unclear": "unclear"}


def load_labels(path: Path) -> dict[int, str]:
    """article_id -> normalised human verdict."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["article_id"]):
                NORM.get((row.get("verdict") or "").strip().lower(), "?")
                for row in csv.DictReader(handle)}

def sent_ids_from_prompt(prompt: str) -> list[int]:
    """Recover, in order, the article_ids that were sent.

    Read out of the saved PROMPT rather than passed in. The prompt is the only
    record of what a given run actually saw, so a run scored years later cannot
    be scored against the wrong input list. It also recovers the order, which
    is what "S1", "S2" citations mean.
    """
    return [int(m) for m in re.findall(r"^article_id: (\d+)$", prompt, re.M)]

def articles_with_incidents(saved: dict[str, Any]) -> tuple[set[int], list[int]]:
    """Return (article_ids that produced an incident, the ids sent).

    Handles both citation styles: early runs cited raw article_ids, run 6
    onward cites "S1" labels. resolve_labels turns labels into ids and leaves
    anything else untouched, so a mistyped id still shows up as itself and is
    counted as a miss rather than silently mapped onto something.
    """
    sent = sent_ids_from_prompt(saved["prompt"])
    reply = saved["response"]["choices"][0]["message"]["content"]
    incidents = resolve_labels(extract_json(reply).get("incidents") or [], sent)
    cited = {s for i in incidents for s in (i.get("sources") or [])
             if isinstance(s, int)}
    return cited & set(sent), sent

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=Path("eval/rohini_labels.csv"))
    parser.add_argument("--responses", type=Path,
                        default=Path("eval/responses"))
    args = parser.parse_args()

    labels = load_labels(args.labels)

    runs = []
    for path in sorted(args.responses.glob("*.json")):
        if ".error-" in path.name:
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not (saved.get("label") or "").startswith("generate:"):
            continue
        try:
            found, sent = articles_with_incidents(saved)
        except (ValueError, KeyError, IndexError):
            continue
        runs.append((saved["requested_at"][11:19], found, sent))

    if not runs:
        raise SystemExit("no generation responses found")

    runs.sort()
    sent = runs[0][2]
    # Only articles the judge called `relevant` belong in the denominator.
    # `unclear` is excluded rather than counted either way -- the judge could
    # not decide, so the model cannot be marked right or wrong on it (F-049).
    should = [a for a in sent if labels.get(a) == "relevant"]
    unclear = [a for a in sent if labels.get(a) == "unclear"]

    print(f"sources sent      : {len(sent)}")
    print(f"labelled relevant : {len(should)}   <- the recall denominator")
    print(f"labelled unclear  : {len(unclear)}   <- excluded from scoring")
    print(f"runs scored       : {len(runs)}\n")

    header = "  ".join(t for t, _, _ in runs)
    print(f"{'article':<11}{'label':<12}{header}   found")
    print("-" * (25 + 10 * len(runs)))

    hits = defaultdict(int)
    for article in sent:
        marks = []
        for _, found, _ in runs:
            hit = article in found
            hits[article] += hit
            marks.append(" Y      " if hit else " .      ")
        flag = "" if labels.get(article) != "relevant" or hits[article] == len(runs) else "  <-"
        print(f"{article:<11}{labels.get(article, '?'):<12}"
              f"{''.join(marks)}{hits[article]}/{len(runs)}{flag}")

    print()
    rates = [len([a for a in should if a in found]) / len(should)
             for _, found, _ in runs]
    for (stamp, _, _), rate in zip(runs, rates):
        print(f"  {stamp}   recall {rate:.2f}")
    print(f"\n  mean {sum(rates)/len(rates):.2f}   "
          f"range {min(rates):.2f}-{max(rates):.2f}")

    print("\nNOTE: labels are per ARTICLE; the generator emits one entry per")
    print("INCIDENT, and one article can hold several (84303161 has two).")
    print("So this is a FLOOR on recall, not an exact figure.")


if __name__ == "__main__":
    main()
