"""Privacy-safe look at reference-detector items that survive anonymisation (dev split only).

Shows, per entity type, whether a surviving item is made of common/medical words (printed: not
identifying, typically a reference-detector false positive) or of name-like / rare tokens (masked).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from medanon import Anonymizer
from medanon.evaluation.diagnostics import safe_surface, shape
from medanon.evaluation.metrics import _CALENDAR_ANCHOR, ReferenceDetector
from medanon.lexicon import load_lexicon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/patient_doctor_encounters.jsonl")
    ap.add_argument("--n", type=int, default=140)
    args = ap.parse_args()
    lex = load_lexicon()
    anon, ref = Anonymizer(), ReferenceDetector()
    recs = [json.loads(line) for line in open(args.dev, encoding="utf-8")][: args.n]
    results, _ = anon.anonymize_dataset(recs)
    surv = Counter()
    for r, (o, _) in zip(recs, results):
        for f in ("dialogue", "note"):
            for etype, surf in set(ref.find(r[f])):
                s = surf.strip()
                if etype == "DATE_TIME" and not _CALENDAR_ANCHOR.search(s):
                    continue
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", o[f], re.I):
                    shown = safe_surface(s)
                    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", s)]
                    cls = shown if shown == s.lower() else (
                        f"NAME-LIKE {shape(s)}" if any(lex.name_ratio(w) >= 0.3 for w in words) else f"other {shape(s)}")
                    surv[(etype, f, cls)] += 1
    for k, v in surv.most_common(80):
        print(f"{v:4d} {k}")


if __name__ == "__main__":
    main()
