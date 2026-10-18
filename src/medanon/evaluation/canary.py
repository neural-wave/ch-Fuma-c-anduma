"""Canary injection: measure recall *in real clinical context* without gold labels.

Synthetic identifiers are inserted into real encounters (new dialogue turns between existing
turns; sentences inside note sections) at known offsets. The anonymiser runs on the injected
record and we measure how many canaries are removed, split by label and by whether the canary
token is covered by our lexicons/gazetteer ("in-vocabulary") or not ("out-of-vocabulary").
Nothing from the real text is ever printed.
"""
from __future__ import annotations

import random
import re
from collections import Counter

from ..lexicon import load_lexicon

OOV_FIRST = ["xiomara", "oluwaseun", "thandiwe", "anneliese", "radoslav", "yerlan", "chidinma", "ileana",
             "bartholomew", "zanele", "quillon", "sigrun", "tavita", "ewelina", "makoa", "jovanka"]
OOV_LAST = ["okonkwo", "przybylski", "vanterpool", "abeywickrama", "szczepanski", "oyelowo", "nakashima",
            "wojciechowski", "haddadin", "ekwueme", "lindqvist", "mbeki", "castellanos", "tarkovsky"]
OOV_CITIES = ["duluth", "scranton", "kalamazoo", "pensacola", "provo", "yuma", "macon", "bozeman", "waco",
              "missoula", "bangor", "eugene", "fargo", "laredo"]
COUNTRIES = ["portugal", "kenya", "vietnam", "peru", "ghana", "norway", "morocco"]
EMPLOYERS = ["fedex", "costco", "lockheed martin", "home depot", "starbucks", "boeing", "kroger", "safeway"]
HOSPITALS = ["mercy hospital", "st. vincent's hospital", "riverside medical center", "baptist health clinic"]
OCCS = ["electrician", "teacher", "truck driver", "accountant", "firefighter", "pharmacist", "chef"]

DIALOGUE_CANARIES = [
    ("[patient] my {rel} , {first} , drove me here today .", {"first": "PERSON"}),
    ("[patient] {first} is going to pick me up later .", {"first": "PERSON"}),
    ("[doctor] i'll send a copy of this to doctor {last} .", {"last": "PERSON"}),
    ("[patient] we just moved here from {city} .", {"city": "LOCATION"}),
    ("[patient] i work at {employer} as a {occ} .", {"employer": "ORGANIZATION", "occ": "OCCUPATION"}),
    ("[patient] they saw me at {hospital} on {date} .", {"hospital": "ORGANIZATION", "date": "DATE"}),
    ("[patient] i'll be {age} next month .", {"age": "AGE"}),
    ("[doctor] can you confirm your date of birth ?\n[patient] it's {dob} .", {"dob": "DOB"}),
    ("[patient] we were in {country} over {holiday} .", {"country": "LOCATION", "holiday": "HOLIDAY"}),
    ("[patient] my phone number is {phone} .", {"phone": "CONTACT"}),
]
NOTE_CANARIES = [
    ("{First} {Last} called on {Date} to reschedule.", {"First": "PERSON", "Last": "PERSON", "Date": "DATE"}),
    ("She is accompanied by her daughter, {First}.", {"First": "PERSON"}),
    ("Works as a {occ} at {Employer} in {City}.", {"occ": "OCCUPATION", "Employer": "ORGANIZATION", "City": "LOCATION"}),
    ("Previously seen by Dr. {Last} at {Hospital}.", {"Last": "PERSON", "Hospital": "ORGANIZATION"}),
]


def _values(rng: random.Random, lex) -> dict:
    oov = rng.random() < 0.4
    firsts = [n for n, p in lex.first_names.items() if p >= 0.05 and lex.name_ratio(n) >= 0.3]
    lasts = [n for n, c in lex.surnames.items() if c >= 20000 and lex.name_ratio(n) >= 0.3]
    first = rng.choice(OOV_FIRST if oov else firsts)
    last = rng.choice(OOV_LAST if oov else lasts)
    month = rng.choice(["january", "march", "april", "june", "august", "october", "november"])
    return {
        "first": first, "last": last, "First": first.capitalize(), "Last": last.capitalize(), "oov": oov,
        "rel": rng.choice(["wife", "husband", "daughter", "son", "sister", "neighbor"]),
        "city": rng.choice(OOV_CITIES), "country": rng.choice(COUNTRIES), "employer": rng.choice(EMPLOYERS),
        "hospital": rng.choice(HOSPITALS), "occ": rng.choice(OCCS), "age": str(rng.randint(25, 88)),
        "date": f"{month} {rng.randint(2, 28)}th", "Date": f"{month.capitalize()} {rng.randint(2, 28)}, 2023",
        "dob": f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(1935, 1999)}",
        "holiday": rng.choice(["christmas", "thanksgiving", "easter"]), "phone": f"555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
    }


def _fill(template: str, slots: dict, vals: dict, offset: int):
    text, spans = "", []
    for piece in re.split(r"(\{\w+\})", template):
        m = re.fullmatch(r"\{(\w+)\}", piece)
        if not m:
            text += piece
            continue
        key = m.group(1)
        val = vals[key.lower()].title() if key in ("Employer", "Hospital", "City") else vals[key]
        if key in slots:
            spans.append({"start": offset + len(text), "end": offset + len(text) + len(val), "label": slots[key],
                          "tpl": template.strip()[:28],
                          "oov": vals["oov"] if slots[key] == "PERSON" else key.lower() in ("city",)})
        text += val
    return text, spans


def inject(record: dict, rng: random.Random, n_dialogue: int = 3, n_note: int = 2) -> tuple[dict, list[dict]]:
    lex = load_lexicon()
    rec = dict(record)
    gold: list[dict] = []
    # dialogue: insert whole turns at line boundaries
    lines = rec["dialogue"].split("\n")
    positions = sorted(rng.sample(range(1, len(lines)), k=min(n_dialogue, max(1, len(lines) - 1))), reverse=True)
    inserts = {p: rng.choice(DIALOGUE_CANARIES) for p in positions}
    vals = _values(rng, lex)
    out_lines, offset = [], 0
    for i, line in enumerate(lines):
        if i in inserts:
            tpl, slots = inserts[i]
            text, spans = _fill(tpl, slots, vals, offset)
            for s in spans:
                s["field"] = "dialogue"
            gold += spans
            out_lines.append(text)
            offset += len(text) + 1
            vals = _values(rng, lex)
        out_lines.append(line)
        offset += len(line) + 1
    rec["dialogue"] = "\n".join(out_lines)
    # note: append sentences to the end of randomly chosen non-empty paragraphs
    note = rec["note"]
    paras = [m for m in re.finditer(r"[^\n]*[a-z][^\n]*[.!?]", note)]
    chosen = sorted(rng.sample(paras, k=min(n_note, len(paras))), key=lambda m: m.end(), reverse=True)
    for m in chosen:
        tpl, slots = rng.choice(NOTE_CANARIES)
        vals = _values(rng, lex)
        text, spans = _fill(" " + tpl, slots, vals, m.end())
        shift = len(text)
        for g in gold:
            if g["field"] == "note" and g["start"] >= m.end():
                g["start"] += shift
                g["end"] += shift
        for s in spans:
            s["field"] = "note"
        gold += spans
        note = note[:m.end()] + text + note[m.end():]
    rec["note"] = note
    return rec, gold


def evaluate(records: list[dict], anonymizer, seed: int = 11, rounds: int = 1) -> dict:
    rng = random.Random(seed)
    tot, hit = Counter(), Counter()
    for _ in range(rounds):
        for record in records:
            rec, gold = inject(record, rng)
            _, decs = anonymizer.anonymize(rec)
            repl = {(d["field"], i) for d in decs if d["action"] != "keep" and d["label"] != "LAB_VALUE"
                    for i in range(d["start"], d["end"])}
            for g in gold:
                text = rec[g["field"]]
                ok = all((g["field"], i) in repl for i in range(g["start"], g["end"]) if text[i].isalnum())
                for key in (g["label"], f"{g['label']}/{g['field']}", f"{g['label']}/{'oov' if g['oov'] else 'iv'}",
                            f"{g['label']}/tpl:{g['tpl']}"):
                    tot[key] += 1
                    hit[key] += ok
                tot["ALL"] += 1
                hit["ALL"] += ok
    return {k: {"n": tot[k], "recall": round(hit[k] / tot[k], 3)} for k in sorted(tot)}
