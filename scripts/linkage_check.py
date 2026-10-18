"""Closed-world demographic linkage on a SYNTHETIC development set (output-only attacker view).

    python scripts/linkage_check.py <input.jsonl> <gold.jsonl> <out1.jsonl> [<out2.jsonl> ...]

Attacker knowledge per target (from the original + gold): exact age, sex, encounter year(s). Released view,
parsed from the anonymised text only: published age (exact "N-year-old", band "[a-b]", "90+"), a surviving
birth year ("[DOB] YYYY" -> age within one year of the reference year), surviving calendar years in dates
("[DATE] YYYY"), sex from pronouns. A target is linked when its candidate set contains exactly one record and
that record is the target. Small closed world, accurate knowledge: an upper-bound stress test, not population risk.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medanon.evaluation.attacks import _sex  # noqa: E402

REF_YEAR = 2026


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def knowledge(rec, gold):
    ages = [int(a["text"]) for a in gold["annotations"] if a["label"] == "AGE" and a["text"].isdigit()]
    years = {int(y) for a in gold["annotations"] if a["label"] == "DATE" for y in re.findall(r"\b(?:19|20)\d{2}\b", a["text"])}
    return {"age": ages[0] if ages else None, "sex": _sex(rec["note"] + " " + rec["dialogue"][:3000]), "years": years}


def released(o):
    text = o["note"] + "\n" + o["dialogue"]
    lo, hi = 0, 200
    if m := re.search(r"(?<![\[\d])(\d{1,3})[- ]year[- ]old", text):
        lo = hi = int(m.group(1))
    elif m := re.search(r"\[(\d+)-(\d+)\]-year-old", text):
        lo, hi = int(m.group(1)), int(m.group(2))
    elif re.search(r"90\+-year-old", text):
        lo, hi = 90, 200
    if m := re.search(r"\[DOB\] ((?:19|20)\d{2})", text):  # birth year: age is REF-y-1 .. REF-y
        y = int(m.group(1))
        lo, hi = max(lo, REF_YEAR - y - 1), min(hi, REF_YEAR - y)
    years = {int(y) for y in re.findall(r"\[DATE\] ((?:19|20)\d{2})", text)}
    return {"lo": lo, "hi": hi, "sex": _sex(o["note"] + " " + o["dialogue"][:3000]), "years": years}


def attack(know, pub):
    res = {"unique_correct": 0, "exact_age_recoverable": 0, "mean_candidates": 0.0}
    for i, k in enumerate(know):
        cand = [j for j, p in enumerate(pub)
                if (k["age"] is None or p["lo"] <= k["age"] <= p["hi"])
                and (k["sex"] == "U" or p["sex"] in (k["sex"], "U"))
                and (not p["years"] or not k["years"] or k["years"] & p["years"])]
        res["unique_correct"] += cand == [i]
        res["mean_candidates"] += len(cand) / len(know)
        res["exact_age_recoverable"] += pub[i]["hi"] - pub[i]["lo"] <= 1
    res["mean_candidates"] = round(res["mean_candidates"], 2)
    return res


def main():
    inp, gold = load(sys.argv[1]), {g["encounter_id"]: g for g in load(sys.argv[2])}
    know = [knowledge(r, gold[r["encounter_id"]]) for r in inp]
    for p in sys.argv[3:]:
        outs = {o["encounter_id"]: o for o in load(p)}
        pub = [released(outs[r["encounter_id"]]) for r in inp]
        print(f"{len(inp)} targets | {Path(p).parent.parent.name}/{Path(p).parent.name}: {attack(know, pub)}")


if __name__ == "__main__":
    main()
