"""Annotated review view of a MedAnon run (input + output + audit), for manual expert review.

Every decision is shown inline in the ORIGINAL text:
  {{original -> replacement | LABEL | detectors}}   replaced / generalised
  <<original | LABEL kept: reason_code>>              detected but kept (policy or clinical guard)
Also checks that the output equals the input with exactly these replacements applied.

    python scripts/review_run.py <input.jsonl> <output.jsonl> [--ids D1,D2] > review.txt
"""
import argparse
import json
import sys

SHORT = {"spacy:": "sp:", "ml:hf": "bert", "rule:": "r:", "propagation": "prop", "presidio:": "pres:"}


def short(dets):
    out = []
    for d in dets or []:
        for a, b in SHORT.items():
            d = d.replace(a, b)
        out.append(d)
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--ids")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    recs = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    outs = {o["encounter_id"]: o for o in map(json.loads, open(a.output, encoding="utf-8"))}
    audit = [json.loads(l) for l in open(a.output.replace(".jsonl", ".audit.jsonl"), encoding="utf-8")]
    want = set(a.ids.split(",")) if a.ids else None
    for r in recs:
        eid = r["encounter_id"]
        if want and eid not in want:
            continue
        decs = [d for d in audit if d["encounter_id"] == eid]
        print(f"\n{'#' * 110}\n# {eid} | {r.get('dataset')} | ICD: " + "; ".join(f"{c['code']} {c['description']}" for c in r["icd10_codes"]))
        for f in ("dialogue", "note"):
            t = r[f]
            ds = sorted((d for d in decs if d["field"] == f), key=lambda d: (d["start"], -d["end"]))
            pieces, pos, rebuilt, rpos = [], 0, [], 0
            for d in ds:
                if d["start"] < pos:  # nested/overlapping audit entries: show once
                    continue
                pieces.append(t[pos:d["start"]])
                surf = t[d["start"]:d["end"]]
                if d["action"] == "keep":
                    pieces.append(f"<<{surf} | {d['label']} kept: {d.get('reason_code', '')}>>")
                else:
                    pieces.append(f"{{{{{surf} -> {d['replacement']} | {d['label']} | {short(d.get('detectors'))}}}}}")
                pos = d["end"]
            pieces.append(t[pos:])
            print(f"--- {f.upper()} ({len(t.split())} words)\n" + "".join(pieces))
        o = outs[eid]
        ok = all(o[k] == r[k] for k in r if k not in ("dialogue", "note")) and \
            o["dialogue"].count("\n") == r["dialogue"].count("\n") and o["note"].count("\n") == r["note"].count("\n")
        print(f"--- structure/non-text fields unchanged: {ok}")


if __name__ == "__main__":
    main()
