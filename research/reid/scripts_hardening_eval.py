"""Paired production-hardening evaluation on synthetic worlds: baseline (ceff209 + age_bands) vs candidate.

    python scripts_hardening_eval.py <runs dir> <out.json> world[:base_policy:cand_policy] ...

For each world: clinical facts kept, residual identifiers (research/reid/reid/clinical.py), structural validity
(record order, keys, icd10_codes, speaker turns, note lines) and section-header preservation. Aggregates only.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid import clinical  # noqa: E402

HEADER = re.compile(r"^[A-Z][A-Z &/]{2,40}:?$|^[A-Z][a-z]+(?: [A-Za-z]+){0,4}:$")


def structure(inp: Path, out: Path) -> dict:
    a = [json.loads(line) for line in open(inp, encoding="utf-8")]
    b = [json.loads(line) for line in open(out, encoding="utf-8")]
    hdr_n = hdr_k = 0
    for x, y in zip(a, b):
        for line in x["note"].split("\n"):
            if HEADER.match(line.strip()):
                hdr_n += 1
                hdr_k += line in y["note"].split("\n")
    return {"records_in": len(a), "records_out": len(b),
            "same_order_ids": [r["encounter_id"] for r in a] == [r["encounter_id"] for r in b],
            "non_text_fields_identical": all({k: v for k, v in x.items() if k not in ("dialogue", "note")} ==
                                             {k: v for k, v in y.items() if k not in ("dialogue", "note")} for x, y in zip(a, b)),
            "section_headers_kept": f"{hdr_k}/{hdr_n}"}


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = {}
    for spec in sys.argv[3:]:
        w, base, cand = (spec.split(":") + ["age_bands", "cand_age_bands"])[:3]
        wd = runs / w
        with gzip.open(wd / "private" / "persons.csv.gz", "rt", encoding="utf-8") as f:
            persons = pd.read_csv(f, low_memory=False)
        for tag, pol in (("base", base), ("cand", cand)):
            o = wd / "anonymized" / pol / "encounters.jsonl"
            res.setdefault(w, {})[tag] = {"policy_dir": pol, "structure": structure(wd / "release_input" / "encounters.jsonl", o),
                                         "clinical_and_residual": clinical.check(wd, o, persons)}
    out.write_text(json.dumps(res, indent=1))
    keys = sorted({k for w in res.values() for k in w["base"]["clinical_and_residual"] if k != "structure"})
    for w, r in res.items():
        print("==", w, "| structure base", r["base"]["structure"], "| cand", r["cand"]["structure"])
        for k in keys:
            b, c = r["base"]["clinical_and_residual"].get(k), r["cand"]["clinical_and_residual"].get(k)
            if b and c:
                print(f"  {k:36} {b['kept']:>4}/{b['n']:<4} -> {c['kept']:>4}/{c['n']:<4}")


if __name__ == "__main__":
    main()
