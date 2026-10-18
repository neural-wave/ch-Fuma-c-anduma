"""Score a MedAnon run against span-level reference annotations (synthetic development packages).

    python scripts/score_gold.py <input.jsonl> <output.jsonl> <gold.jsonl> [--audit A] [--json report.json] [--show N]

Reuses medanon.evaluation.scoring.score: privacy targets = every gold label except CLINICAL; CLINICAL spans
(negations, doses, lab results, family-history onset ages ...) are "traps" that must survive unchanged.
Raw recall counts deliberately published ages/years as misses; policy-compliant recall is reported separately.
--show N prints up to N examples per error class - only for SYNTHETIC data (it prints original text).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medanon.evaluation.scoring import score  # noqa: E402


# policy-sensitive quasi-identifiers (not conventional PHI): reported as separate objectives, outside the
# core privacy denominator, per (label, kind, clinical_importance) as "share transformed"
POLICY_LABELS = ("ANTHROPOMETRIC", "FAMILY_SIZE", "FAMILY_RELATION")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("gold")
    ap.add_argument("--audit")
    ap.add_argument("--json")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    global args_show
    args_show = a.show
    sys.stdout.reconfigure(encoding="utf-8")
    recs = {r["encounter_id"]: r for r in load(a.input)}
    outs = {o["encounter_id"]: o for o in load(a.output)}
    audit = load(a.audit or a.output.replace(".jsonl", ".audit.jsonl"))
    gold = {g["encounter_id"]: g for g in load(a.gold)}
    ids = sorted(gold)
    by_id = defaultdict(list)
    for d in audit:
        by_id[d["encounter_id"]].append(d)
    R, O, D = [], [], []
    for i in ids:
        r = dict(recs[i])
        r["_gold"] = [dict(g, hard=False) for g in gold[i]["annotations"] if g["label"] not in ("CLINICAL", *POLICY_LABELS)]
        r["_policy"] = [g for g in gold[i]["annotations"] if g["label"] in POLICY_LABELS]
        r["_traps"] = [dict(g, kind=g.get("clinical_kind", "clinical")) for g in gold[i]["annotations"] if g["label"] == "CLINICAL"]
        R.append(r), O.append(outs[i]), D.append(by_id[i])
    res = score(R, O, D)
    # error lists: missed / partially removed gold entities, clinical traps damaged, transformations outside gold
    miss, partial, traps, fp = Counter(), Counter(), Counter(), Counter()
    ex = defaultdict(list)
    for r, o, decs in zip(R, O, D):
        repl = defaultdict(set)
        for d in decs:
            if d["action"] != "keep" and d["label"] != "LAB_VALUE":
                repl[d["field"]].update(range(d["start"], d["end"]))
        gold_chars = defaultdict(set)
        for g in r["_gold"]:
            gold_chars[g["field"]].update(range(g["start"], g["end"]))
            alnum = [k for k in range(g["start"], g["end"]) if r[g["field"]][k].isalnum()]
            n = sum(k in repl[g["field"]] for k in alnum)
            key = (g["label"], g.get("role", ""))
            if n == len(alnum):
                continue
            (partial if n else miss)[key] += 1
            ex[("partial" if n else "miss", g["label"])].append(f"{r['encounter_id']} {g['field']}: {g['text']!r}")
        for t in r["_traps"]:
            if any(k in repl[t["field"]] for k in range(t["start"], t["end"])):
                traps[t["kind"]] += 1
                ex[("trap", t["kind"])].append(f"{r['encounter_id']}: {t['text']!r}")
        for d in decs:
            if d["action"] == "keep" or d["label"] == "LAB_VALUE":
                continue
            if not any(k in gold_chars[d["field"]] for k in range(d["start"], d["end"])):
                fp[d["label"]] += 1
                ex[("fp", d["label"])].append(f"{d['encounter_id']} {d['field']}: {r[d['field']][d['start']:d['end']]!r} -> {d['replacement']}")
    res["errors"] = {"missed": {f"{k[0]}{'/' + k[1] if k[1] else ''}": v for k, v in miss.items()},
                     "partially_removed": {f"{k[0]}{'/' + k[1] if k[1] else ''}": v for k, v in partial.items()},
                     "clinical_traps_damaged": dict(traps), "transformations_outside_gold": dict(fp)}
    # strict token proxy (NOT Bosch's metric): an entity counts as removed only if none of its tokens (>= 2
    # alphanumerics) is still present anywhere in the output field, so "[DOB] 1985" or a kept age is not removed
    import re
    st_tot, st_ok = Counter(), Counter()
    for r, o in zip(R, O):
        toks = {f: set(re.findall(r"[a-z0-9]{2,}", o[f].lower())) for f in ("dialogue", "note")}
        for g in r["_gold"]:
            left = [t for t in re.findall(r"[a-z0-9]{2,}", g["text"].lower()) if t in toks[g["field"]]]
            st_tot[g["label"]] += 1
            st_ok[g["label"]] += not left
    res["strict_token_removal"] = {"ALL": round(sum(st_ok.values()) / max(1, sum(st_tot.values())), 4),
                                   **{k: round(st_ok[k] / st_tot[k], 3) for k in sorted(st_tot)}}
    pol_n, pol_t = Counter(), Counter()
    for r, decs in zip(R, D):
        repl = defaultdict(set)
        for d in decs:
            if d["action"] != "keep":
                repl[d["field"]].update(range(d["start"], d["end"]))
        for g in r["_policy"]:
            key = f"{g['label']}/{g.get('kind', '-')}/{g.get('clinical_importance', '-')}"
            pol_n[key] += 1
            hit = any(k in repl[g["field"]] for k in range(g["start"], g["end"]) if r[g["field"]][k].isalnum())
            pol_t[key] += hit
            if args_show and len(ex[("policy", key)]) < args_show:
                ex[("policy", key)].append(f"{'TRANSFORMED' if hit else 'kept'} {r['encounter_id']}: {g['text']!r}")
    res["policy_objectives_share_transformed"] = {k: f"{pol_t[k]}/{pol_n[k]}" for k in sorted(pol_n)}
    res["structure_ok"] = sum(all(o[k] == r[k] for k in ("encounter_id", "dataset", "icd10_codes")) and
                              all(o[f].count("\n") == r[f].count("\n") for f in ("dialogue", "note")) for r, o in zip(R, O))
    keys = ("records", "gold_entities", "recall", "policy_compliant_recall", "precision", "f1", "kept_by_policy",
            "encounters_with_direct_identifier_left", "trap_preservation_all", "structure_ok", "strict_token_removal", "policy_objectives_share_transformed",
            "errors")
    print(json.dumps({k: res[k] for k in keys}, indent=1))
    print("recall by label:", {k[6:]: v["value"] for k, v in res["recall_by"].items() if k.startswith("label:")})
    if a.show:
        for k in sorted(ex):
            print(f"--- {k[0]} {k[1]} ({len(ex[k])})")
            for e in ex[k][:a.show]:
                print("   ", e)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
