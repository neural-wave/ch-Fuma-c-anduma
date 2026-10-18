"""Pre-publication privacy check of the repository text (run before every push).

Derives high-precision patient-name seeds from the local datasets (names after a title, subjects of
"X is a N-year-old" in the notes) and checks:
  1. documentation (*.md, configs): no seed token at all                                   -> FAIL
  2. quoted examples in the docs: none occurs verbatim in the real data with a rare word,
     unless it also occurs in our synthetic generators / tests                            -> FAIL
  3. tracked code, tests, scripts and commit messages: a seed after a title, or a rare
     seed (zipf < 3.5) anywhere                                                           -> WARN (review by hand)
Only file:line and masked context are printed, never a real token.

    python scripts/privacy_check_docs.py                  # working tree; exit code 1 if any FAIL
    python scripts/privacy_check_docs.py --ref origin/x   # what is already published on a branch
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from medanon.lexicon import load_lexicon

ROOT = Path(__file__).resolve().parents[1]
TITLE = re.compile(r"\b(?:mr|mrs|ms|miss|dr)\.?\s+([a-z'\-]+)", re.I)
WORD = re.compile(r"[A-Za-z'\-]+")
SNIPPET = re.compile(r'"([^"\n]{6,120})"|`([^`\n]{6,120})`|“([^”\n]{6,120})”')
RARE_ZIPF = 3.5


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_real(paths: list[Path]) -> tuple[set[str], str]:
    seeds, blobs = set(), []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            note = r.get("note", "")
            seeds |= {w.lower() for w in re.findall(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-z'\-]+)", note)}
            for m in re.finditer(r"(?:^|\n)((?:[A-Z][a-z'\-]+\s){0,2}[A-Z][a-z'\-]+) is an? \d+-year-old", note):
                seeds |= {w.lower() for w in m.group(1).split() if w.lower() not in {"the", "patient", "this"}}
            blobs.append(_norm(r.get("dialogue", "")) + " \n " + _norm(note))
    return seeds, "\n".join(blobs)


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def repo_files(ref: str | None) -> list[str]:
    if ref:
        files = _git("ls-tree", "-r", "--name-only", ref).split("\n")
    else:
        files = _git("ls-files").split("\n") + _git("ls-files", "--others", "--exclude-standard").split("\n")
    return sorted({f for f in files if f and not f.startswith("src/medanon/resources/")})


def read(f: str, ref: str | None) -> str:
    return _git("show", f"{ref}:{f}") if ref else (ROOT / f).read_text(encoding="utf-8")


def mask(line: str, spans: list[tuple[int, int]]) -> str:
    out, last = "", 0
    for a, b in sorted(spans):
        out += line[last:a] + "<<masked>>"
        last = b
    return (out + line[last:]).strip()[:200]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[ROOT / "data/patient_doctor_encounters.jsonl", ROOT / "data/eval_patient_doctor_encounters.jsonl"])
    ap.add_argument("--ref", help="check files and commit messages at this git ref instead of the working tree")
    args = ap.parse_args()
    seeds, real_blob = load_real(args.data)
    lex = load_lexicon()
    files = repo_files(args.ref)
    docs = [f for f in files if f.endswith(".md") or (f.startswith("configs/") and f.endswith((".yaml", ".yml", ".json")))]
    code = [f for f in files if f.endswith((".py", ".toml", ".txt", ".cfg")) and f not in docs]
    synthetic = _norm(" ".join(read(f, args.ref) for f in files
                               if f.startswith(("src/medanon/evaluation/", "tests/")) and f.endswith(".py")))
    fails = warns = 0

    for f in docs:
        for i, line in enumerate(read(f, args.ref).splitlines(), 1):
            hits = [(m.start(), m.end()) for m in WORD.finditer(line) if m.group().lower() in seeds]
            if hits:
                fails += len(hits)
                print(f"FAIL name seed      {f}:{i}: {mask(line, hits)}")
            for m in SNIPPET.finditer(line):
                s = _norm(next(g for g in m.groups() if g))
                if len(s.split()) < 2 or s in synthetic or s not in real_blob:
                    continue
                if any(lex.zipf(w) < RARE_ZIPF for w in re.findall(r"[a-z']+", s)):
                    fails += 1
                    print(f"FAIL real example   {f}:{i}: {mask(line, [m.span()])}")

    def scan_code(name: str, text: str) -> int:
        n = 0
        for i, line in enumerate(text.splitlines(), 1):
            hits = [m.span(1) for m in TITLE.finditer(line) if m.group(1).lower() in seeds]
            hits += [m.span() for m in WORD.finditer(line)
                     if m.group().lower() in seeds and lex.zipf(m.group().lower()) < RARE_ZIPF and m.span() not in hits]
            if hits:
                n += len(hits)
                print(f"WARN review by hand {name}:{i}: {mask(line, hits)}")
        return n

    for f in code:
        try:
            warns += scan_code(f, read(f, args.ref))
        except UnicodeDecodeError:
            continue
    warns += scan_code("<commit messages>", _git("log", args.ref or "--all", "--format=%B"))

    print(f"{len(seeds)} name seeds | {len(docs)} docs, {len(code)} code files checked | FAIL {fails} | WARN {warns}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
