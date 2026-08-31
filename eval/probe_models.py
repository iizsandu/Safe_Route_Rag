"""Probe free models with a two-word prompt to see which are alive.

F-050 cost 8 wasted calls by sending a 50,000-character job to an endpoint
that was already dead, and its recorded lesson was that three cheap probes
would have cost 3 calls and saved 8. This is that lesson as a script.

On 2026-08-28 the pinned model `nvidia/nemotron-3.5-lightning:free` began
returning 504 on every request -- including "Say OK" -- while OpenRouter
still LISTED it as available. Exactly F-050's failure shape.

Run:
    python -m eval.probe_models
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag.llm import ModelConfig, complete

# Domain-specific models are excluded on purpose: a finance-tuned or
# coding-tuned model is a worse prior for Indian crime reporting, whatever
# its benchmark scores. The current pinned model is included so we can see
# whether it recovers rather than assuming it has not.
CANDIDATES = [
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling-small:free",
    "dots-studio/dots-3-note-preview:free",
    "thinkingmachines/inkling:free",
]

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for model in CANDIDATES:
        try:
            reply = complete("Say OK", ModelConfig(model=model),
                             Path("eval/responses"), label="probe")
            print(f"  ALIVE  {model:<45} {reply.text[:60]!r}")
        except SystemExit as error:
            # complete() raises SystemExit on a failed call. Catching it here
            # is the point -- one dead model must not end the probe run.
            first = str(error).splitlines()[0]
            print(f"  DEAD   {model:<45} {first}")

if __name__ == "__main__":
    main()
