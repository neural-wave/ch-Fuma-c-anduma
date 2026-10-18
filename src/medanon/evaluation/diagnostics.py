"""Privacy-safe diagnostics on real records (development split).

Never prints raw text. A surface string is shown only when it is *not* identifying - a common
English word or a medical term (exactly the cases that indicate false positives); everything
else is reduced to a shape such as 'Xxxxx' / 'dd/dd/dddd'.
"""
from __future__ import annotations

import re
from collections import Counter

from ..lexicon import load_lexicon


def shape(s: str) -> str:
    return re.sub(r"[a-z]", "x", re.sub(r"[A-Z]", "X", re.sub(r"\d", "d", s)))[:24]


def safe_surface(s: str) -> str:
    """Return the string if every word is common/medical (non-identifying), else its shape."""
    lex = load_lexicon()
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
    if re.search(r"\b(?:mr|mrs|ms|miss|dr|doctor)\b", s, re.I):
        return shape(s)  # a title marks the rest as a personal name, however common the word
    if words and all((lex.zipf(w.lower()) >= 4.0 and lex.name_ratio(w.lower()) < 0.05) or lex.medical_strength(w.lower()) == 2
                     for w in words):
        return s.lower()
    return shape(s)


def diagnose(records: list[dict], anonymizer, top: int = 25) -> dict:
    lex = load_lexicon()
    by_label, by_src, kept = Counter(), Counter(), Counter()
    suspicious, medical_hits = Counter(), Counter()
    changed_chars = total_chars = 0
    residual = Counter()
    for rec in records:
        out, decs = anonymizer.anonymize(rec)
        for d in decs:
            text = rec[d["field"]]
            surface = text[d["start"]:d["end"]]
            if d["action"] == "keep":
                kept[(d.get("candidate_label", d["label"]), safe_surface(surface))] += 1
                continue
            by_label[(d["field"], d["label"], d["action"])] += 1
            for src in d["detectors"]:
                by_src[(d["label"], src.split(":")[0] + ":" + src.split(":")[-1])] += 1
            if d["label"] not in ("LAB_VALUE",):
                changed_chars += d["end"] - d["start"]
                shown = safe_surface(surface)
                if shown == surface.lower() and d["label"] not in ("AGE", "DATE", "DOB", "YEAR", "ID", "MONTH", "WEEKDAY", "HOLIDAY", "OCCUPATION"):
                    suspicious[(d["label"], shown, ",".join(sorted({s.split(":")[0] for s in d["detectors"]})))] += 1
                for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", surface):
                    if lex.medical_strength(w.lower()) == 2:
                        medical_hits[(d["label"], w.lower())] += 1
        for fld in ("dialogue", "note"):
            total_chars += len(rec[fld])
            # residual capitalised mid-sentence tokens in the anonymised note that are neither common,
            # medical nor placeholders: potential missed proper nouns (shape only)
            if fld == "note":
                for m in re.finditer(r"(?<=[a-z,;]\s)([A-Z][a-z]{2,})", out[fld]):
                    w = m.group(1).lower()
                    if lex.zipf(w) < 3.5 and lex.medical_strength(w) == 0 and w not in lex.icd_vocab:
                        residual["name-like" if lex.in_name_lexicon(w) else "other"] += 1
    print(f"records={len(records)} changed_char_ratio={changed_chars / max(1, total_chars):.4f}")
    print("\n# replacements by field/label/action"); [print(f"  {v:5d} {k}") for k, v in sorted(by_label.items())]
    print("\n# detector contributions"); [print(f"  {v:5d} {k}") for k, v in by_src.most_common(40)]
    print(f"\n# replaced spans made only of common/medical words (likely false positives), top {top}")
    [print(f"  {v:4d} {k}") for k, v in suspicious.most_common(top)]
    print("\n# medical-lexicon words inside replaced spans"); [print(f"  {v:4d} {k}") for k, v in medical_hits.most_common(top)]
    print(f"\n# kept by medical guard, top {top}"); [print(f"  {v:4d} {k}") for k, v in kept.most_common(top)]
    print("\n# residual capitalised rare tokens in anonymised notes (potential misses):", dict(residual))
    return {"by_label": by_label, "suspicious": suspicious, "residual": residual}


if __name__ == "__main__":
    import argparse
    import json

    from ..pipeline import Anonymizer

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config")
    args = ap.parse_args()
    with open(args.input, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    diagnose(recs[: args.limit or None], Anonymizer(config_path=args.config))
