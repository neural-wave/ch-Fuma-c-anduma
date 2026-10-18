"""LLM client: JSON-schema output over the OpenAI-compatible or native Ollama API, with a
disk cache, retries and a thread pool."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from collections.abc import Callable

import requests

log = logging.getLogger("attack.llm")

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# The spec default was qwen2.5:32b "(check /api/tags for what's actually pulled)"; the GPU host has qwen3.5:35b.
DEFAULT_MODEL = "qwen3.5:35b"
SEED = 42
RETRY_TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 4096  # extraction answers use ~700 tokens, recovery ~500


class LLMError(RuntimeError):
    """A request that failed after all retries, or a server we cannot use."""


class ContextOverflow(LLMError):
    """The prompt did not fit in num_ctx; retrying cannot help."""


def inline_refs(schema: dict) -> dict:
    """Inline pydantic's $defs/$ref and drop titles: grammar converters prefer flat schemas."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, backend: str = "auto",
                 num_ctx: int = 16384, cache_dir: str = "attack/cache", timeout: float = 900.0,
                 retries: int = 3):
        self.base_url = (base_url or os.environ.get("ATTACK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.root = self.base_url.removesuffix("/v1")
        self.model = model or os.environ.get("ATTACK_MODEL") or DEFAULT_MODEL
        self.backend = backend
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retries = retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"llm_calls": 0, "cache_hits": 0, "retries": 0}
        self.thinking = False  # Ollama model with a thinking mode: disabled, the JSON is the answer
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Check the server and model, then pick the backend. On Ollama, `auto` uses the native
        /api/chat: the OpenAI-compatible endpoint cannot set num_ctx, and Ollama silently
        truncates prompts longer than its default context (our records reach ~6k tokens)."""
        tags = _get_json(f"{self.root}/api/tags")
        is_ollama = bool(tags) and isinstance(tags.get("models"), list)
        if is_ollama:
            models = [m.get("name") for m in tags["models"]]
        else:
            listing = _get_json(f"{self.base_url}/models")
            if not listing or not isinstance(listing.get("data"), list):
                raise LLMError(f"no LLM API at {self.base_url}; run `python -m attack.check_endpoint`")
            models = [m.get("id") for m in listing["data"]]
        if self.model not in models and f"{self.model}:latest" not in models:
            raise LLMError(f"model {self.model!r} is not available; pulled: {', '.join(models)}")
        if self.backend == "auto":
            self.backend = "ollama" if is_ollama else "openai"
        if is_ollama:
            try:
                show = requests.post(f"{self.root}/api/show", json={"model": self.model}, timeout=30).json()
                self.thinking = "thinking" in (show.get("capabilities") or [])
            except (requests.RequestException, ValueError):
                pass
        log.info("LLM model=%s backend=%s base=%s", self.model, self.backend, self.base_url)

    def chat_json(self, messages: list[dict], schema: dict, name: str, prompt_version: str,
                  validate: Callable[[dict], dict]) -> dict:
        """Structured completion; `validate` raises ValueError on bad output and returns the
        normalized dict. Results are cached by sha256(model + prompt version + prompt)."""
        prompt_text = json.dumps({"messages": messages, "schema": schema}, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(f"{self.model}\n{prompt_version}\n{prompt_text}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self._count("cache_hits")
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self._count("retries")
                time.sleep(2 ** (attempt - 1))
            try:
                # Greedy decoding (temperature 0) replays the same bad output whatever the seed, so retries
                # sample at RETRY_TEMPERATURE with a shifted seed: different, and still reproducible.
                temperature = 0.0 if attempt == 0 else RETRY_TEMPERATURE
                data = validate(json.loads(self._call(messages, schema, name, SEED + attempt, temperature)))
            except ContextOverflow:
                raise
            except (requests.RequestException, ValueError, KeyError, LLMError) as e:
                error = e
                log.warning("%s attempt %d/%d failed: %s", name, attempt + 1, self.retries + 1, str(e)[:200])
                continue
            tmp = path.with_name(f"{key}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"model": self.model, "prompt_version": prompt_version,
                                       "response": data}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            return data
        raise LLMError(f"{name}: gave up after {self.retries + 1} attempts: {error}")

    def _call(self, messages: list[dict], schema: dict, name: str, seed: int, temperature: float = 0.0) -> str:
        self._count("llm_calls")
        if self.backend == "openai":
            body = {"model": self.model, "messages": messages, "temperature": temperature, "seed": seed,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "response_format": {"type": "json_schema",
                                        "json_schema": {"name": name, "schema": schema, "strict": True}}}
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
            if r.status_code in (400, 422) and "format" in r.text.lower():
                log.warning("server rejected response_format; falling back to Ollama /api/chat")
                self.backend = "ollama"
                return self._call(messages, schema, name, seed, temperature)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError("output truncated (finish_reason=length)")
            return choice["message"]["content"]

        body = {"model": self.model, "messages": messages, "stream": False, "format": schema, "keep_alive": "30m",
                "options": {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx,
                            "num_predict": MAX_OUTPUT_TOKENS}}
        if self.thinking:
            body["think"] = False
        r = requests.post(f"{self.root}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("prompt_eval_count", 0) >= self.num_ctx - MAX_OUTPUT_TOKENS:
            raise ContextOverflow(f"prompt alone fills num_ctx={self.num_ctx}; raise --num-ctx")
        if data.get("done_reason") == "length" or data.get("eval_count", 0) >= MAX_OUTPUT_TOKENS:
            # runaway output (e.g. a repetition loop): retryable, the retry samples differently
            raise LLMError(f"output hit the {MAX_OUTPUT_TOKENS}-token cap (done_reason=length)")
        return data["message"]["content"]

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1


def _get_json(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def run_pool(fn: Callable[[Any], Any], items: list, workers: int = 8, desc: str = "tasks") -> list:
    """Run fn over items in a thread pool; a failed item yields None instead of stopping the run."""
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                results[futures[future]] = future.result()
            except Exception as e:  # logged and skipped, never fatal
                log.error("%s item %d skipped: %s", desc, futures[future], e)
            if done % 10 == 0 or done == len(items):
                log.info("%s: %d/%d", desc, done, len(items))
    return results
