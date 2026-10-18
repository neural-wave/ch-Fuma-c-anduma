"""Deterministic mask audit (no LLM): what did each placeholder replace, and what identifying
residue is still visible? Complements the LLM attack with exact, reproducible counts.

    python -m attack.audit_masks --original orig.jsonl --anonymized anon.jsonl --out audit.json

- Alignment (needs the original; scorer side only): every placeholder is mapped to the original
  span it replaced and the span is classified (measurement, clinical term, date, name-like, other).
  A placeholder over a measurement or clinical term is over-redaction.
- Residue scans (anonymized text only, i.e. what an attacker sees): exact adult ages, bare numeric
  answers to age questions, visible birth years (4- and 2-digit), age expressions priced in bits.
`audit.json` holds counts only; `--examples FILE` also writes spans and contexts for local review
(they contain original text: never share them).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher

from . import tables
from .data import PLACEHOLDER_RE, load_jsonl
from .recover import _TOKEN
from .verify import MEDICAL_RE

BP_RE = re.compile(r"^\d{2,3}\s*(?:/|over)\s*\d{2,3}$", re.I)
NUMBER_UNIT_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:%|percent|mg|mcg|g|ml|mmhg|bpm|kg|lbs?|pounds?|cm|mm|units?)?$", re.I)
CLINICAL_RE = re.compile(r"\b(?:lungs?|heart|abdomen|neuro\w*|vascular|covid|vaccines?|biofreeze|icy ?hot|tylenol|"
                         r"advil|motrin|aleve|ibuprofen|cream|ointment|splint|brace|exam|reflex\w*|pulses?|"
                         r"murmur|extremit\w+|cervical|lumbar|thoracic|knee|elbow|shoulder|ankle|wrist)\b", re.I)
DATE_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(?:mon|tues|wednes|thurs|"
                     r"fri|satur|sun)day\b|\b\d{1,2}(?:st|nd|rd|th)\b|^\d{1,2}(?:\s+\d{1,4}){0,2}$|\b(?:19|20)\d{2}\b",
                     re.I)
NAME_LIKE_RE = re.compile(r"^(?:(?:mr|mrs|ms|miss|dr|doctor)\s+)?[a-z][a-z'’]+(?:\s+[a-z][a-z'’]+){0,2}$", re.I)

ADULT_AGE_RE = re.compile(r"(?<![\[\d-])\b(1[89]|[2-9]\d|1[01]\d)\s*-?\s*(?:(?:years?|yrs?)\s*-?\s*old|y/?o)\b"
                          r"|\b(?:aged?|age of)\s+(1[89]|[2-9]\d|1[01]\d)\b(?![\d.%-])", re.I)
AGE_QUESTION_RE = re.compile(r"how old are you|how old you are|your age\b|what age are you", re.I)  # the patient only
BARE_ADULT_NUMBER_RE = re.compile(r"(?<![\[\d./-])\b(1[89]|[2-9]\d)\b(?![\d./%-]|\s*(?:mg|%|percent|pounds|lbs?|kg))")
_DOB = r"(?:\[DOB\]|\b(?:born|birth|dob|d\.o\.b\.?)\b)"  # includes the "[DOB] 1982" residue form
BIRTH_YEAR4_RE = re.compile(_DOB + r"[^.\n]{0,40}?\b((?:19|20)\d{2})\b", re.I)
BIRTH_YEAR2_RE = re.compile(_DOB + r"[^.\n]{0,40}?(?:'|’|/|-|\bin\s|\]\s*)(\d{2})\b(?![\d/])", re.I)
AGE_EXPR_RE = re.compile(r"(\[\d{1,3}-\d{1,3}\]|\b\d{1,3}\+?|\[\d{1,3}\+\])\s*-?\s*(?:years?|yrs?)\s*-?\s*old", re.I)


def replacements(anon_text: str, orig_text: str) -> list[dict]:
    """One entry per aligned 'replace' block that contains a placeholder."""
    a, o = _TOKEN.findall(anon_text), _TOKEN.findall(orig_text)
    sm = SequenceMatcher(None, [t.lower() for t in a], [t.lower() for t in o], autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        phs = [t for t in a[i1:i2] if PLACEHOLDER_RE.fullmatch(t)]
        if tag == "replace" and phs:
            span = " ".join(o[j1:j2])
            out.append({"placeholders": phs, "original_span": span, "category": classify(span, phs),
                        "context": " ".join(a[max(0, i1 - 6):i2 + 6])})
    return out


IDENTIFIER_LABELS = {"ID", "CONTACT", "ADDRESS", "DOB", "PATIENT_NAME", "CLINICIAN_NAME", "RELATIVE_NAME"}


def classify(span: str, placeholders: list[str] = ()) -> str:
    s = span.strip()
    if any(re.match(r"\[([A-Z_]*?[A-Z])(?:_\d+)?[\]:]", p).group(1) in IDENTIFIER_LABELS for p in placeholders):
        return "identifier"
    if BP_RE.match(s):
        return "measurement_bp"
    if NUMBER_UNIT_RE.match(s) and not DATE_RE.search(s):
        return "measurement_number"
    if MEDICAL_RE.search(s) or CLINICAL_RE.search(s):
        return "clinical_term"
    if DATE_RE.search(s):
        return "date_time"
    if NAME_LIKE_RE.match(s):
        return "name_like"
    return "other"


def residue(anon_text: str) -> dict:
    lines = anon_text.splitlines()
    age_answers = []
    for i, line in enumerate(lines[:-1]):
        if AGE_QUESTION_RE.search(line):
            nxt = lines[i + 1]
            if m := BARE_ADULT_NUMBER_RE.search(nxt):
                age_answers.append(m.group(1))
    ages = [m.group(1) or m.group(2) for m in ADULT_AGE_RE.finditer(anon_text)]
    age_bits = [_age_bits(m.group(1)) for m in AGE_EXPR_RE.finditer(anon_text)]
    return {"exact_adult_ages": ages, "age_question_answers": age_answers,
            "birth_years_4digit": BIRTH_YEAR4_RE.findall(anon_text),
            "birth_years_2digit": BIRTH_YEAR2_RE.findall(anon_text),
            "age_bits_max": max(age_bits, default=0.0)}


def _age_bits(expr: str) -> float:
    rng = tables.parse_age(expr.strip("[]"))
    return -math.log2(tables.age_range_share(*rng)) if rng else 0.0


def audit(orig_path: str, anon_path: str) -> tuple[dict, list[dict]]:
    orig = {r["encounter_id"]: r for r in load_jsonl(orig_path)}
    rows, examples = [], []
    for r in load_jsonl(anon_path):
        o = orig.get(r["encounter_id"])
        a_text = (r.get("dialogue") or "") + "\n" + (r.get("note") or "")
        res = residue(a_text)
        reps = []
        if o:
            for field in ("dialogue", "note"):
                reps += [dict(x, field=field) for x in replacements(r.get(field) or "", o.get(field) or "")]
        rows.append({"encounter_id": r["encounter_id"], "residue": res,
                     "replacement_categories": Counter(x["category"] for x in reps)})
        examples += [dict(x, encounter_id=r["encounter_id"]) for x in reps
                     if x["category"] in ("measurement_bp", "measurement_number", "clinical_term")]
        examples += [{"encounter_id": r["encounter_id"], "residue": k, "values": v}
                     for k, v in res.items() if k != "age_bits_max" and v]
    n = len(rows)

    def share(key: str) -> dict:
        hit = [x["encounter_id"] for x in rows if x["residue"][key]]
        return {"records": len(hit), "of": n, "ids": hit}

    cats = Counter()
    for x in rows:
        cats.update(x["replacement_categories"])
    over_cats = ("measurement_bp", "measurement_number", "clinical_term")
    over = [x["encounter_id"] for x in rows if any(x["replacement_categories"].get(c) for c in over_cats)]
    summary = {
        "n_records": n,
        "replacement_blocks_by_category": dict(cats.most_common()),
        "records_with_overredaction": {"records": len(over), "of": n, "ids": over},
        "exact_adult_age_visible": share("exact_adult_ages"),
        "bare_age_answer_after_age_question": share("age_question_answers"),
        "birth_year_4digit_visible": share("birth_years_4digit"),
        "birth_year_2digit_visible": share("birth_years_2digit"),
        "mean_age_bits_max": sum(x["residue"]["age_bits_max"] for x in rows) / n if n else None,
    }
    return summary, examples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m attack.audit_masks", description=__doc__.splitlines()[0])
    ap.add_argument("--original", required=True)
    ap.add_argument("--anonymized", required=True)
    ap.add_argument("--out", required=True, help="counts-only JSON (shareable)")
    ap.add_argument("--examples", help="JSONL with original spans and contexts (local review only)")
    args = ap.parse_args(argv)
    summary, examples = audit(args.original, args.anonymized)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"original": args.original, "anonymized": args.anonymized, **summary}, f, indent=2)
    if args.examples:
        with open(args.examples, "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) or "ids" not in v else f"{v['records']}/{v['of']}")
                      for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
