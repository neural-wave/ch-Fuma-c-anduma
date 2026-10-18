"""Step 0: find a working Ollama / OpenAI-compatible endpoint behind the SSH tunnel.

Usage: python -m attack.check_endpoint [--chat-test] [--model qwen3.5:35b]

Stdlib only, so it runs before any dependency is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROBES = [
    "http://localhost:8080/v1/models",
    "http://localhost:8080/api/tags",
    "http://localhost:11434/v1/models",
    "http://localhost:11434/api/tags",
]


def probe(url: str, timeout: float = 5.0) -> dict:
    res = {"url": url, "ok": False, "kind": None, "models": [], "detail": ""}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            body = r.read(200_000).decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        res["detail"] = f"HTTP {e.code}"
        return res
    except Exception as e:  # connection refused, timeout, ...
        res["detail"] = f"{type(e).__name__}: {e}"
        return res

    if "html" in ctype.lower() or body.lstrip().startswith("<"):
        res["kind"] = "html"
        res["detail"] = f"HTTP {status}, HTML page (looks like Open WebUI, not the API)"
        return res
    try:
        data = json.loads(body)
    except ValueError:
        res["detail"] = f"HTTP {status}, non-JSON body ({ctype})"
        return res

    if url.endswith("/v1/models") and isinstance(data.get("data"), list):
        res.update(ok=True, kind="openai", models=[m.get("id") for m in data["data"]])
    elif url.endswith("/api/tags") and isinstance(data.get("models"), list):
        res.update(ok=True, kind="ollama", models=[m.get("name") for m in data["models"]])
    else:
        res["detail"] = f"HTTP {status}, unexpected JSON keys {list(data)[:5]}"
    return res


def chat_test(base_url: str, model: str) -> str:
    """One tiny request. On Ollama it goes through the native /api/chat with thinking disabled, as
    attack.run does: through /v1, a thinking model spends a small max_tokens budget on reasoning and
    returns an empty reply."""
    root = base_url.rstrip("/").removesuffix("/v1")
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    if _is_json(root + "/api/tags"):
        # Same num_ctx as attack.run: a different context size makes Ollama reload the model.
        options = {"temperature": 0, "seed": 42, "num_predict": 5, "num_ctx": 16384}
        url, payload = root + "/api/chat", {"model": model, "messages": messages, "stream": False, "think": False,
                                            "options": options}
    else:
        url, payload = base_url.rstrip("/") + "/chat/completions", {
            "model": model, "messages": messages, "temperature": 0, "seed": 42, "max_tokens": 5}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError:  # e.g. an old Ollama rejecting "think" for a non-thinking model
        if "think" not in payload:
            raise
        payload.pop("think")
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(payload).encode(),
                                                           headers={"Content-Type": "application/json"}),
                                    timeout=600) as r:
            data = json.loads(r.read())
    msg = data["message"] if "message" in data else data["choices"][0]["message"]
    reply = (msg.get("content") or "").strip()
    note = "" if reply else "  WARNING: empty reply (a thinking model spending the budget on reasoning?)"
    return f"{reply!r} via {url.removeprefix(root)} in {time.time() - t0:.1f}s{note}"


def _is_json(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            json.loads(r.read())
        return True
    except (OSError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chat-test", action="store_true", help="also send a tiny chat completion")
    ap.add_argument("--model", default=os.environ.get("ATTACK_MODEL", "qwen3.5:35b"))
    args = ap.parse_args()

    results = [probe(u) for u in PROBES]
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        info = f"{r['kind']}: {len(r['models'])} models" if r["ok"] else r["detail"]
        print(f"[{mark}] {r['url']:<36} {info}")

    by_port = {}
    for r in results:
        port = r["url"].split(":")[2].split("/")[0]
        by_port.setdefault(port, {})[r["url"].rsplit("/", 1)[-1]] = r

    working = [p for p, rs in by_port.items() if rs["models"]["ok"]]
    html_8080 = any(r["kind"] == "html" for r in by_port["8080"].values())

    if not working:
        print("\nNo OpenAI-compatible API found.")
        if html_8080:
            print("Port 8080 serves HTML (Open WebUI). Forward Ollama's own port instead, e.g.:\n"
                  "  ssh -N -L 11434:localhost:11434 <user>@<gpu-host>")
        else:
            print("Is the SSH tunnel up? e.g.  ssh -N -L 11434:localhost:11434 <user>@<gpu-host>")
        return 1

    port = "11434" if "11434" in working else working[0]
    base_url = f"http://localhost:{port}/v1"
    tags = by_port[port]["tags"]
    models = tags["models"] if tags["ok"] else by_port[port]["models"]["models"]
    print(f"\nUse: ATTACK_BASE_URL={base_url}")
    print("Pulled models:", ", ".join(models) or "(none)")
    if args.model not in models:
        print(f"WARNING: default model {args.model!r} is not pulled; pass --model / ATTACK_MODEL.")
    if html_8080:
        print("Note: port 8080 is Open WebUI (HTML); ignore it.")
    if args.chat_test:
        try:
            print("Chat test:", chat_test(base_url, args.model))
        except Exception as e:
            print(f"Chat test FAILED: {type(e).__name__}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
