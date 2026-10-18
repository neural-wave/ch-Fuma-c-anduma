"""Assembles the MedAnon handoff package from an explicit allowlist (no data, caches, results with
record text, or model weights) and writes MANIFEST.json with sha256 and versions.

    .venv/bin/python attack/handoff/build_package.py        # -> dist/medanon_attacker_handoff/ + .zip
"""
import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HANDOFF = REPO / "attack" / "handoff"
NAME = "medanon_attacker_handoff"
OUT = REPO / "dist" / NAME

ATTACK_FILES = ["__init__.py", "audit_masks.py", "check_endpoint.py", "data.py", "export_predictions.py", "extract.py",
                "llm.py", "recover.py", "report.py", "run.py", "score.py", "tables.py", "verify.py",
                "requirements.txt", "ruff.toml", "README.md", "docs/validation5_top_leaking_attributes.png"]
TOP_LEVEL = ["CLAUDE_HANDOFF.md", "VULNERABILITIES.json", "FOR_CLAUDE_CODE.txt"]
SYNTH = ["README.md", "make_fixtures.py", "check_regressions.py", "encounters.jsonl", "expectations.json"]
RESULTS = "results"  # aggregate JSON only, written by the analysis step


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for f in TOP_LEVEL:
        copy(HANDOFF / f, OUT / f)
    for f in ATTACK_FILES:
        copy(REPO / "attack" / f, OUT / "attack" / f)
    for f in sorted((REPO / "attack" / "tests").glob("test_*.py")):
        copy(f, OUT / "attack" / "tests" / f.name)
    for f in SYNTH:
        copy(HANDOFF / "synthetic_regressions" / f, OUT / "synthetic_regressions" / f)
    for f in sorted((HANDOFF / "synthetic_regressions" / "outputs").glob("*.json*")):
        copy(f, OUT / "synthetic_regressions" / "outputs" / f.name)
    for f in sorted((HANDOFF / RESULTS).glob("*.json")):
        copy(f, OUT / RESULTS / f.name)
    # the offline end-to-end test reads the fixtures from attack/handoff/synthetic_regressions
    for f in SYNTH:
        copy(HANDOFF / "synthetic_regressions" / f, OUT / "attack" / "handoff" / "synthetic_regressions" / f)

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    versions = {}
    for pkg in ("requests", "pydantic", "rapidfuzz", "matplotlib", "pytest", "ruff", "medanon", "spacy", "torch",
                "transformers", "presidio-analyzer"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    provenance = json.loads((HANDOFF / RESULTS / "provenance.json").read_text())
    files = {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    manifest = {
        "package": NAME,
        "attacker_repo": "progetto-hackaton, branch adversary-reidentification",
        "attacker_commit": git("rev-parse", "HEAD"),
        "working_tree_clean": git("status", "--porcelain", "--", "attack") == "",
        "medanon_commit_tested": "ceff209 (src/medanon and configs/ unchanged on this branch)",
        "medanon_code_sha256_local": provenance["medanon_code_sha256_local"],
        "python": platform.python_version(),
        "packages": versions,
        "llm": {"model": "qwen3.5:35b", "digest": "72aac7c28be8ff9b81c2a270eb489c88e6d822a76ec0cb7bf34d965768848567",
                "quantization": "Q4_K_M", "server": "Ollama 0.34.0 on RTX 4090 D 48 GB",
                "temperature": 0, "seed": "42 (+attempt on retry)", "num_ctx": 16384, "think": False,
                "prompts": {"extract": "extract-v4", "recover": "recover-v3"}},
        "files_sha256": files,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    zpath = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, Path(NAME) / p.relative_to(OUT))
    print(f"{len(files) + 1} files -> {zpath} ({zpath.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
