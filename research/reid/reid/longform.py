"""Long-form synthetic stress world: each record concatenates 3-4 renderings of the SAME fictional patient
(different Synthea encounters, dev template family), giving task-like lengths (dialogue ~6-7k, note ~3-4k
characters) with identifiers repeated across turns and fields. Same layout as reid.world (private/ truth).

    python -m reid.longform --synthea <pop csv dir> --out <world dir> --seed 11 --n 60
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import pandas as pd

from .render import render_encounter
from .world import augment, build_encounter, hash_seed, load_population


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthea", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args(argv)
    csv, out = Path(a.synthea), Path(a.out)
    popn = load_population(csv, csv.parent / "population.pkl")
    persons = popn["persons"]
    augment(persons, seed=hash_seed(",".join(sorted(persons)[:50])))
    rng = random.Random(a.seed)
    pids = sorted(p for p, es in popn["encs"].items() if len(es) >= 3 and persons[p]["age_ref"] >= 18)
    rng.shuffle(pids)
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "release_input").mkdir(parents=True, exist_ok=True)
    recs, truth = [], []
    for pid in pids[:a.n]:
        p = persons[pid]
        es = sorted(rng.sample(popn["encs"][pid], rng.choice([3, 4])), key=lambda e: e["date"])
        dias, notes, facts, exp = [], [], [], {}
        for e in es:
            enc = build_encounter(p, e, popn, rng)
            d, n, f, x = render_encounter(p, enc, 0, rng)
            dias.append(d)
            notes.append(n)
            facts += f
            for k, v in x.items():
                exp[k] = exp.get(k, []) + v if isinstance(v, list) else (exp.get(k) or v)
        last = build_encounter(p, es[-1], popn, rng)
        eid = "".join(rng.choice("0123456789abcdef") for _ in range(12))
        recs.append({"encounter_id": eid, "dataset": "synthetic-long", "dialogue": "\n".join(dias),
                     "note": "\n\n".join(notes[:2]), "icd10_codes": [{"code": c, "description": ""} for c in last["icd"]]})
        n_note = sum(1 for f in facts if f["field"] == "note")
        # facts of the 3rd/4th notes are not in the record (only two notes are kept)
        note_facts_kept = [f for f in facts if f["field"] == "dialogue"] + \
            [f for f in facts if f["field"] == "note" and any(str(f.get("surface", f.get("term", ""))) in nn for nn in notes[:2])]
        truth.append({"encounter_id": eid, "pid": pid, "visit_index": 0, "date": es[-1]["date"].isoformat(),
                      "age": last["age"], "primary_icd": last["primary_icd"], "reason_kind": last["reason_kind"],
                      "facts": note_facts_kept, "exposure": exp, "n_note_facts_total": n_note})
    with open(out / "release_input" / "encounters.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(out / "private" / "encounters_truth.jsonl", "w", encoding="utf-8") as f:
        for t in truth:
            f.write(json.dumps(t) + "\n")
    cols = ["pid", "first", "last", "sex", "birthdate", "race", "marital", "city", "occupation", "occ_status",
            "work_pattern", "children", "hobby", "height", "weight", "bmi", "age_ref"]
    with gzip.open(out / "private" / "persons.csv.gz", "wt", encoding="utf-8") as f:
        pd.DataFrame([{c: persons[t["pid"]].get(c) for c in cols} for t in truth]).to_csv(f, index=False)
    (out / "private" / "manifest.json").write_text(json.dumps({"seed": a.seed, "family": "dev", "n_encounters": len(recs),
                                                               "kind": "longform", "synthea_csv": str(csv)}))
    L = [(len(r["dialogue"]), len(r["note"])) for r in recs]
    print(len(recs), "records; median dialogue", sorted(x[0] for x in L)[len(L) // 2], "note", sorted(x[1] for x in L)[len(L) // 2])


if __name__ == "__main__":
    main()
