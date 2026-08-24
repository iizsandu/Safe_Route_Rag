"""Score an LLM filter's verdicts against a human-labelled answer key.

Reads saved API responses from eval/responses/ -- never calls an API. Scoring
is cheap and re-runnable; the responses are the expensive artifact. That split
is deliberate: a bug in here costs nothing to fix and re-run.

Run:
    python eval/score_filter.py --model nvidia/nemotron-3.5-lightning:free
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# The human labels arrived with three spellings for the same verdict. Rather
# than editing the ground truth to suit the code, the code normalises on read.
# Label vocabularies always need this; the answer key should stay as the judge
# wrote it.
VERDICT_ALIASES = {
    "relevant": "relevant",
    "not relevant": "not_relevant",
    "not_relevant": "not_relevant",
    "irrelevant": "not_relevant",
    "unclear": "unclear",
}

POSITIVE = "relevant"


def normalise(verdict: str) -> str:
    """Map a written verdict onto the canonical vocabulary."""
    return VERDICT_ALIASES.get((verdict or "").strip().lower(), "?")


def load_labels(path: Path) -> dict[int, dict[str, str]]:
    """Read the answer key, preserving row order via the returned dict."""
    labels: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            article_id = int(row["article_id"])
            labels[article_id] = {
                "verdict": normalise(row.get("verdict", "")),
                "category": (row.get("category") or "").strip(),
                "note": (row.get("note") or "").strip(),
            }
    return labels


def extract_verdicts(reply: str) -> list[dict[str, Any]]:
    """Pull the JSON array of verdicts out of a model reply.

    Models wrap JSON in prose, in ```json fences, or emit it bare. Rather than
    demanding one shape, take the outermost [...] span and parse that. Returns
    an empty list if nothing parses -- the caller reports it as missing rather
    than crashing, so one malformed batch cannot hide the other three.
    """
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def load_predictions(
    directory: Path,
    model: str,
    prompt_contains: str | None = None,
    prompt_omits: str | None = None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Collect this model's verdicts, newest run wins. Returns (verdicts, runs).

    Ordered by `requested_at`, NOT by filename. Filenames are content hashes,
    so filename order is arbitrary -- sorting by it silently blended a Friday
    run with a Monday run and produced a score belonging to neither. Prompt
    variants are selected with prompt_contains / prompt_omits so an A/B
    comparison can be scored without deleting anything.
    """
    records = []
    for path in directory.glob("*.json"):
        if ".error-" in path.name:
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("model") != model:
            continue
        prompt = saved.get("prompt") or ""
        if prompt_contains and prompt_contains not in prompt:
            continue
        if prompt_omits and prompt_omits in prompt:
            continue
        if not ((saved.get("response") or {}).get("choices") or []):
            continue
        records.append((saved.get("requested_at") or "", path, saved))

    predictions: dict[int, dict[str, str]] = {}
    runs: set[str] = set()
    for requested_at, path, saved in sorted(records, key=lambda r: r[0]):
        reply = (saved["response"]["choices"][0].get("message") or {}).get("content") or ""
        for item in extract_verdicts(reply):
            try:
                article_id = int(item["article_id"])
            except (KeyError, TypeError, ValueError):
                continue
            predictions[article_id] = {
                "verdict": normalise(item.get("verdict", "")),
                "reason": (item.get("reason") or "").strip(),
                "batch": path.name,
                "requested_at": requested_at,
            }
    for prediction in predictions.values():
        runs.add(prediction["requested_at"])
    return predictions, sorted(runs)


def load_headlines(path: Path) -> dict[int, str]:
    """Headlines, for readable output only -- never used in scoring."""
    if not path.is_file():
        return {}
    headlines = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                headlines[record["article_id"]] = record.get("headline") or ""
    return headlines


def report(
    labels: dict[int, dict[str, str]],
    predictions: dict[int, dict[str, str]],
    headlines: dict[int, str],
    model: str,
) -> None:
    """Print coverage, the per-article comparison, and the summary metrics."""
    missing = [a for a in labels if a not in predictions]
    print(f"model     : {model}")
    print(f"labelled  : {len(labels)} articles")
    print(f"predicted : {len(predictions)} articles")
    if missing:
        print(f"MISSING   : {len(missing)} -> {missing}")
        print("  A missing verdict is not a wrong answer. Investigate before")
        print("  reading any metric below.")

    print(f"\n{'article':<11}{'key':<14}{'model':<14}   headline")
    print("-" * 96)
    agree = 0
    disagreements = []
    for article_id, label in labels.items():
        prediction = predictions.get(article_id)
        predicted = prediction["verdict"] if prediction else "MISSING"
        ok = predicted == label["verdict"]
        agree += ok
        print(f"{article_id:<11}{label['verdict']:<14}{predicted:<14}"
              f"{'ok ' if ok else 'X  '}{headlines.get(article_id, '')[:48]}")
        if not ok and prediction:
            disagreements.append((article_id, label, prediction))

    total = len(labels)
    print("-" * 96)
    print(f"agreement : {agree}/{total} = {100 * agree / total:.0f}%")

    # Confusion, treating "relevant" as the positive class.
    counts = Counter()
    for article_id, label in labels.items():
        prediction = predictions.get(article_id)
        if not prediction:
            continue
        actual = label["verdict"] == POSITIVE
        guessed = prediction["verdict"] == POSITIVE
        counts["tp" if actual and guessed else
               "fp" if guessed else
               "fn" if actual else "tn"] += 1

    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    print(f"\ntreating '{POSITIVE}' as positive:")
    print(f"  TP {tp}   FP {fp}   FN {fn}   TN {tn}")
    if tp + fp:
        precision = tp / (tp + fp)
        print(f"  precision {precision:.2f}   (of those it kept, how many were right)")
    if tp + fn:
        recall = tp / (tp + fn)
        print(f"  recall    {recall:.2f}   (of the real ones, how many it kept)")
    if tp and (tp + fp) and (tp + fn):
        f1 = 2 * precision * recall / (precision + recall)
        print(f"  F1        {f1:.2f}")

    unclear = sum(1 for p in predictions.values() if p["verdict"] == "unclear")
    print(f"\nmodel said 'unclear' {unclear} times")

    if disagreements:
        print(f"\n{'=' * 96}\nDISAGREEMENTS -- read the article before assuming the model is wrong")
        for article_id, label, prediction in disagreements:
            print(f"\n{article_id}  {headlines.get(article_id, '')[:70]}")
            print(f"  key   : {label['verdict']:<14} {label['note'][:60]}")
            print(f"  model : {prediction['verdict']:<14} {prediction['reason'][:60]}")

    print(f"\nNOTE: {total} articles from one neighbourhood is a smoke test, not a")
    print("measurement. Confidence intervals at this n span tens of points.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=Path("eval/rohini_labels.csv"),
                        help="CSV answer key")
    parser.add_argument("--responses", type=Path,
                        default=Path("eval/responses"),
                        help="Directory of saved API responses")
    parser.add_argument("--sample", type=Path,
                        default=Path("eval/rohini_sample20.jsonl"),
                        help="Sample file, used only for headlines")
    parser.add_argument("--model", required=True,
                        help="Exact model string to score")
    parser.add_argument("--prompt-contains", default=None,
                        help="Only score responses whose prompt contains this "
                             "text -- use it to isolate one prompt variant")
    parser.add_argument("--prompt-omits", default=None,
                        help="Only score responses whose prompt does NOT "
                             "contain this text (the other side of an A/B)")
    args = parser.parse_args()

    if not args.labels.is_file():
        raise SystemExit(f"Not a file: {args.labels}")
    if not args.responses.is_dir():
        raise SystemExit(f"Not a directory: {args.responses}")

    labels = load_labels(args.labels)
    predictions, runs = load_predictions(
        args.responses, args.model, args.prompt_contains, args.prompt_omits
    )
    if not predictions:
        raise SystemExit(f"No saved responses match model {args.model}")

    # Batches within one invocation are seconds apart; separate experiments are
    # not. Warn on a wide span, not on any difference -- a guard that cries wolf
    # gets ignored, which is worse than no guard.
    span_hours = 0.0
    if len(runs) > 1:
        first = datetime.fromisoformat(runs[0])
        last = datetime.fromisoformat(runs[-1])
        span_hours = (last - first).total_seconds() / 3600

    if span_hours > 1:
        print("WARNING: verdicts span", f"{span_hours:.1f} hours --", runs[0][:16], "to", runs[-1][:16])
        print("  This score belongs to no single experiment. Narrow it with")
        print("  --prompt-contains / --prompt-omits before reading anything.")
        print("")
    else:
        print(f"run       : {runs[0][:19]}"
              + (f" .. {runs[-1][11:19]}" if len(runs) > 1 else ""))

    report(labels, predictions, load_headlines(args.sample), args.model)


if __name__ == "__main__":
    main()
