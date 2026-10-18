"""Run manifest: exactly which code, configuration, models and environment produced an output.

Written next to every anonymised file so that the submitted output can be matched to the
benchmarked configuration (no silent fallbacks, no unknown model versions).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from importlib import metadata
from pathlib import Path

PKG_DIR = Path(__file__).parent


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_NOT_CODE = {"__pycache__", "env", ".env", "venv", ".venv", "site-packages"}  # git-ignored environments


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".txt", ".tsv"} and not set(p.relative_to(PKG_DIR).parts) & _NOT_CODE:
            h.update(str(p.relative_to(PKG_DIR)).replace("\\", "/").encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def build_manifest(anonymizer, **extra) -> dict:
    cfg = anonymizer.cfg
    ml = anonymizer.ml
    manifest = {
        "medanon_version": _version("medanon"),
        "code_sha256": code_hash(),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "config": cfg,
        "detectors": {
            "rules": True,
            "spacy": {"enabled": bool(anonymizer.spacy), "model": cfg["spacy"]["model"],
                      "model_version": _version(cfg["spacy"]["model"]), "presidio_patterns": cfg["spacy"]["presidio_patterns"]},
            "ml": {"configured": bool(cfg["ml"].get("enabled")), "status": anonymizer.ml_status,
                   "kind": cfg["ml"].get("kind"), "model": cfg["ml"].get("model"), "mode": cfg["ml"].get("mode"),
                   "fields": cfg["ml"].get("fields"), "revision": getattr(ml, "revision", None),
                   "weights_sha256": ml.weights_sha256() if ml else None, "device": getattr(ml, "device", None)},
            "extra": [{"kind": ec.get("kind"), "model": ec.get("model"), "mode": ec.get("mode"), "fields": ec.get("fields"),
                       "status": "active", "revision": getattr(det, "revision", None),
                       "weights_sha256": det.weights_sha256() if hasattr(det, "weights_sha256") else None,
                       "device": getattr(det, "device", None)}
                      for ec, det in getattr(anonymizer, "extra", [])],
        },
        "lab_perturbation_key": anonymizer.transformer.secret_source,
        "k_anonymity": cfg.get("k_anonymity"),
        "versions": {p: _version(p) for p in ("spacy", "presidio-analyzer", "torch", "transformers", "huggingface-hub")},
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count()},
        "gpu_used": any(getattr(d, "device", "cpu") != "cpu"
                        for d in [ml, *(det for _, det in getattr(anonymizer, "extra", []))] if d is not None),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest.update(extra)
    return manifest


def file_facts(path: str | Path) -> dict:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += bool(line.strip())
    return {"path": str(path), "sha256": _sha256_file(path), "records": n}


class PeakMemory:
    """Samples resident memory of this process *and its children* (worker processes)."""

    def __init__(self, interval: float = 0.25):
        import psutil

        self.proc, self.interval, self.peak = psutil.Process(), interval, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        total = 0
        try:
            total = self.proc.memory_info().rss
            for c in self.proc.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except Exception:  # noqa: BLE001 - child may have exited
                    pass
        except Exception:  # noqa: BLE001
            pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._sample())
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._sample())

    @property
    def peak_mb(self) -> int:
        return round(self.peak / 1e6)
