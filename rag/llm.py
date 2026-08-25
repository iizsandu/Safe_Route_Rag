"""One place that calls an LLM and caches the answer.

Deliberately knows nothing about crime, articles or prompts -- it takes a
string and returns a string. Both the precision filter and the generator use
it, so there is one implementation of caching, error handling and streaming
rather than two that drift.

The raw response is written to disk BEFORE anything is parsed. At roughly 50
free calls a day the response is the expensive artifact; parsing is cheap.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 600

def reply_text(body: dict[str, Any]) -> str:
    """Pull the assistant's text out of an OpenAI-shaped response."""
    choices = body.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""

def is_usable(body: dict[str, Any]) -> bool:
    """True only if the body carries an actual model reply.

    Providers return application-level failures with HTTP status 200 and an
    `error` key instead of `choices` -- a gateway timeout arrives as
    {"error": {"message": "error code: 524"}}. urllib raises nothing for those,
    so a failed call otherwise looks like an empty success, and caching one
    would poison the cache permanently.
    """
    if body.get("error"):
        return False
    return bool(body.get("choices")) and bool(reply_text(body).strip())

def describe_error(body: dict[str, Any]) -> str:
    """Render whatever the provider said went wrong."""
    error = body.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = (error.get("message") or "").strip()
        return f"provider error (code {code}): {message}"
    if error:
        return f"provider error: {error}"
    return "response contained no usable reply and no error message"

@dataclass(frozen=True)
class ModelConfig:
    """Everything about a request except the prompt itself."""

    model: str
    max_tokens: int = 16000
    temperature: int = 0
    reasoning: dict[str, Any] = field(default_factory=lambda: {"effort": "none"})

    def signature(self) -> str:
        """Every knob that changes the output, as a stable string.

        Folded into the cache key. Without it, raising max_tokens would serve a
        reply truncated under the old limit, forever.
        """
        return json.dumps(
            {"max_tokens": self.max_tokens,
             "temperature": self.temperature,
             "reasoning": self.reasoning},
            sort_keys=True,
        )

def response_path(directory: Path, config: ModelConfig, prompt: str) -> Path:
    """Where this exact (model, tuning, prompt) triple's response lives."""
    digest = hashlib.sha256(
        f"{config.model}\n{config.signature()}\n{prompt}".encode("utf-8")
    ).hexdigest()[:12]
    slug = config.model.replace("/", "-").replace(":", "-")
    return directory / f"{slug}__{digest}.json"

def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Does NOT overwrite variables already set, so an exported value always wins.
    Surrounding quotes are stripped: KEY="abc" must yield abc, not "abc" --
    otherwise the quotes travel with the key and the API returns a 401 that
    looks like a bad credential.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))

@dataclass
class Completion:
    """One model reply, and where it came from."""

    text: str
    body: dict[str, Any]
    path: Path
    cached: bool

    @property
    def usage(self) -> dict[str, Any]:
        return self.body.get("usage") or {}

    @property
    def finish_reason(self) -> str | None:
        return (self.body.get("choices") or [{}])[0].get("finish_reason")

def complete(
    prompt: str,
    config: ModelConfig,
    cache_dir: Path,
    label: str = "",
    force: bool = False,
) -> Completion:
    """Return the model's reply, from cache when possible.

    Raises SystemExit on any failure, after writing the failed response to a
    timestamped sidecar. A failure must never land at the cache path, or it
    would be served as an answer forever.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = response_path(cache_dir, config, prompt)

    if destination.exists() and not force:
        saved = json.loads(destination.read_text(encoding="utf-8"))
        body = saved.get("response") or {}
        if is_usable(body):
            return Completion(reply_text(body), body, destination, cached=True)

    load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not found in the environment or in .env")

    try:
        body = _stream(config, prompt, api_key)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code} from the provider:\n{detail}")
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach the provider: {error.reason}")

    usable = is_usable(body)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = destination if usable else (
        destination.parent / f"{destination.stem}.error-{stamp}.json"
    )
    target.write_text(
        json.dumps({"model": config.model,
                    "label": label,
                    "tuning": json.loads(config.signature()),
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "response": body}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not usable:
        raise SystemExit(f"CALL FAILED -- {describe_error(body)}\n  saved: {target}")

    return Completion(reply_text(body), body, target, cached=False)

def _stream(config: ModelConfig, prompt: str, api_key: str) -> dict[str, Any]:
    """Stream the completion, reassembled into a non-streaming response shape.

    Streaming keeps the connection alive while a slow model works, via SSE
    comment lines the provider sends for exactly that purpose. The returned
    dict mirrors a normal completion so callers need no knowledge of it.
    """
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning": config.reasoning,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )

    pieces: list[str] = []
    finish_reason = None
    usage: dict[str, Any] = {}

    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                return {"error": event["error"]}
            for choice in event.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    pieces.append(text)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]

    return {
        "choices": [{"message": {"role": "assistant", "content": "".join(pieces)},
                     "finish_reason": finish_reason}],
        "usage": usage,
    }

    



