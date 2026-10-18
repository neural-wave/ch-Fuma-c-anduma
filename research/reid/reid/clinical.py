"""Evaluator-side residual-information and clinical-utility checks (original vs anonymised text, same records).

Residual identifiers: does the patient's exact first/last name, birth date, city, occupation title survive?
Clinical facts: every clinical element the generator wrote (with its surface) is looked up in the output.
Substring checks are conservative proxies: a rephrased but correct fact counts as lost.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _near(text: str, a: str, b: str, win: int = 90) -> bool:
    tl = text.lower()
    for m in re.finditer(re.escape(a.lower()), tl):
        if b.lower() in tl[max(0, m.start() - win):m.end() + win]:
            return True
    return False


def check(world: Path, policy_out: Path, persons: pd.DataFrame) -> dict:
    orig = {r["encounter_id"]: r for r in map(json.loads, open(world / "release_input" / "encounters.jsonl", encoding="utf-8"))}
    anon = {r["encounter_id"]: r for r in map(json.loads, open(policy_out, encoding="utf-8"))}
    truth = [json.loads(line) for line in open(world / "private" / "encounters_truth.jsonl", encoding="utf-8")]
    per = persons.set_index("pid")
    res = defaultdict(lambda: [0, 0])
    struct = defaultdict(int)
    for t in truth:
        o, a = orig[t["encounter_id"]], anon.get(t["encounter_id"])
        if a is None:
            struct["missing_record"] += 1
            continue
        struct["records"] += 1
        struct["icd_identical"] += a.get("icd10_codes") == o.get("icd10_codes")
        struct["keys_identical"] += set(a) == set(o)
        struct["speaker_turns_identical"] += a["dialogue"].count("[doctor]") + a["dialogue"].count("[patient]") == \
            o["dialogue"].count("[doctor]") + o["dialogue"].count("[patient]")
        struct["note_lines_identical"] += a["note"].count("\n") == o["note"].count("\n")
        p = per.loc[t["pid"]]
        ot, at = o["note"] + "\n" + o["dialogue"], a["note"] + "\n" + a["dialogue"]
        # ---- residual quasi-/direct identifiers (only where the original text contained them)
        for name, val in (("first_name", p["first"]), ("last_name", p["last"]), ("city", p["city"]),
                          ("occupation_title", p["occupation"])):
            if isinstance(val, str) and len(val) > 2 and re.search(rf"\b{re.escape(val)}\b", ot, re.I):
                res[f"residual:{name}"][1] += 1
                res[f"residual:{name}"][0] += bool(re.search(rf"\b{re.escape(val)}\b", at, re.I))
        bd = pd.Timestamp(p["birthdate"])
        for fmt in (f"{bd.month:02d}/{bd.day:02d}/{bd.year}", f"{bd.month}/{bd.day}/{str(bd.year)[2:]}"):
            if fmt in ot:
                res["residual:full_birth_date"][1] += 1
                res["residual:full_birth_date"][0] += fmt in at
        if str(bd.year) in ot:
            res["residual:birth_year_string"][1] += 1
            res["residual:birth_year_string"][0] += str(bd.year) in at
        age = t["age"]
        if re.search(rf"\b{age}(?:-| )?(?:year|y/o)", ot):
            res["residual:exact_age_" + ("child" if age < 18 else "adult")][1] += 1
            res["residual:exact_age_" + ("child" if age < 18 else "adult")][0] += bool(re.search(rf"(?<![\[\d-]){age}(?:-| )?(?:year|y/o)", at))
        # ---- clinical facts
        for f in t["facts"]:
            fld = a[f["field"]]
            k = f["type"]
            if k in ("diagnosis", "medication", "duration", "vital"):
                ok = f["surface"].lower() in fld.lower()
            elif k in ("event_year", "event_date"):
                ok = _near(fld, f["term"], str(f["year"]))
            elif k == "lab":
                ok = _near(fld, f["name"], f["value"], 40)
                res["lab_category_kept"][1] += 1
                res["lab_category_kept"][0] += ok or bool(re.search(re.escape(f["name"]) + r"[^\n;]{0,25}\[", fld, re.I))
            elif k == "anthro":
                ok = f["surface"] in fld
                k = f"anthro_{f['kind']}_" + ("needed" if f["needed"] else "not_needed")
            elif k == "peds_age":
                ok = bool(re.search(rf"(?<![\[\d-]){f['value']}(?:-| )?(?:year|y/o)", fld))
            else:
                continue
            res[f"clinical:{k}"][1] += 1
            res[f"clinical:{k}"][0] += bool(ok)
    out = {k: {"kept": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None} for k, v in sorted(res.items())}
    out["structure"] = dict(struct)
    return out
