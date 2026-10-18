"""Privacy-safe analysis of what an ML detector adds on real dev records.

For spans replaced *only because of* the ML detector, print aggregate categories: label, a
surface class (common/medical words are shown verbatim because they are not identifying;
name-lexicon tokens and rare tokens are masked), and the common-word context around them.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from medanon import Anonymizer, load_config
from medanon.evaluation.diagnostics import safe_surface, shape
from medanon.lexicon import load_lexicon


def ctx_words(text: str, start: int, end: int, lex) -> str:
    left = re.findall(r"[a-z']+|[,.?]", text[max(0, start - 30):start].lower())[-2:]
    right = re.findall(r"[a-z']+|[,.?]", text[end:end + 30].lower())[:2]
    mask = lambda w: w if (lex.zipf(w) >= 4.5 and lex.name_ratio(w) < 0.05) or not w.isalpha() else "<w>"
    return " ".join(map(mask, left)) + " __ " + " ".join(map(mask, right))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/patient_doctor_encounters.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ml", default="gliner:knowledgator/gliner-pii-base-v1.0")
    ap.add_argument("--mode", default="full")
    args = ap.parse_args()
    lex = load_lexicon()
    kind, _, model = args.ml.partition(":")
    anon = Anonymizer(load_config(overrides={"ml": {"enabled": True, "kind": kind, "model": model, "fields": ["dialogue"], "mode": args.mode}}))
    recs = [json.loads(line) for line in open(args.dev, encoding="utf-8")][: args.n]
    by_class, by_ctx = Counter(), Counter()
    for r in recs:
        texts, spans, _, _ = anon.detect(r)
        for s in spans:
            srcs = s.meta.get("sources", [s.source])
            if not all(x.startswith("ml:") for x in srcs):
                continue
            surface = texts[s.field][s.start:s.end]
            shown = safe_surface(surface)
            words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", surface)]
            if shown == surface.lower():
                cls = f"common/medical: {shown}"
            elif words and all(lex.in_name_lexicon(w) for w in words):
                cls = f"name-lexicon tokens x{len(words)} ({shape(surface)})"
            else:
                cls = f"rare/other ({shape(surface)})"
            by_class[(s.label, cls)] += 1
            by_ctx[(s.label, ctx_words(texts[s.field], s.start, s.end, lex))] += 1
    print("# ML-only spans by label / surface class")
    for k, v in by_class.most_common(60):
        print(f"{v:4d}  {k}")
    print("\n# ML-only spans by label / context")
    for k, v in by_ctx.most_common(40):
        print(f"{v:4d}  {k}")


if __name__ == "__main__":
    main()
