"""Before/after check of two MedAnon runs on the same SYNTHETIC input (counts of known issue classes).

    python scripts/compare_run25.py <input.jsonl> <before/out.jsonl> <after/out.jsonl>

Only for synthetic data: it reads the original surfaces the BEFORE run removed and checks whether they are
still present in each output (privacy non-regression), and prints the counts per issue class.
"""
import json
import re
import sys
from collections import Counter

CLINICAL = ["bradykinesia", "Parkinsonian", "canalith", "SpO2", "mIU/L", "Reassess", "Seek urgent care", "Urgent care"]
TEXT = ("dialogue", "note")
SKIP_LABELS = {"LAB_VALUE", "AGE", "DATE", "DOB", "OCCUPATION", "ETHNICITY", "RELATIVE_DATE", "YEAR", "MONTH",
               "WEEKDAY", "HOLIDAY"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def sent_punct(line):
    """Sentence punctuation only: followed by whitespace / end, not the period of a title (Dr., Mr.)."""
    line = re.sub(r"\b(?:Dr|Mr|Mrs|Ms|St)\.", "", line)
    return len(re.findall(r"[.,;:!?](?=\s|$)", line))


def issues(recs, outs):
    c, recs_hit = Counter(), 0
    for r, o in zip(recs, outs):
        hit = False
        for f in TEXT:
            t, a = r[f], o[f]
            for term in CLINICAL:
                lost = t.count(term) - a.count(term)
                if lost > 0:
                    c[f"clinical term lost: {term}"] += lost
                    hit = True
            g = len(re.findall(r"\][A-Za-z0-9]|[A-Za-z0-9]\[", a))
            if g:
                c["placeholder glued to a word (partial-token replacement)"] += g
                hit = True
            for lt, la in zip(t.split("\n"), a.split("\n")):
                lost = sent_punct(lt) - sent_punct(la)
                if lost > 0:
                    c["sentence punctuation lost"] += lost
                    hit = True
            for pat, name in ((r"called \[PERSON\]", "'called [PERSON]' (business named as a person)"),
                              (r"(?:number|it is|it's) \[ID\]", "phone read out -> [ID]"),
                              (r"at \[LOCATION\] today", "clinic -> [LOCATION]"),
                              (r"\ba \[EMPLOYER\]", "generic workplace -> [EMPLOYER]")):
                k = len(re.findall(pat, a))
                if k:
                    c[name] += k
                    hit = True
        if not (all(o[k] == r[k] for k in r if k not in TEXT) and all(o[k].count("\n") == r[k].count("\n") for k in TEXT)):
            c["STRUCTURE CHANGED"] += 1
        recs_hit += hit
    return c, recs_hit


def still_present(audit, rec_by, out_by):
    leaks, checked = Counter(), 0
    for d in audit:
        if d["action"] == "keep" or d["label"] in SKIP_LABELS:
            continue
        surf = rec_by[d["encounter_id"]][d["field"]][d["start"]:d["end"]].strip(" .,;:")
        if len(surf) < 3:
            continue
        checked += 1
        o = out_by[d["encounter_id"]]
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(surf) + r"(?![A-Za-z0-9])", o["dialogue"] + "\n" + o["note"]):
            leaks[d["label"]] += 1
    return leaks, checked


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    inp, before, after = sys.argv[1:4]
    recs = load(inp)
    rec_by = {r["encounter_id"]: r for r in recs}
    res = {}
    for name, p in (("before", before), ("after", after)):
        out_by = {o["encounter_id"]: o for o in load(p)}
        c, n = issues(recs, [out_by[r["encounter_id"]] for r in recs])
        res[name] = (c, n, json.load(open(p.replace(".jsonl", ".manifest.json"), encoding="utf-8")), out_by)
    print(f"{'issue class (occurrences)':58s} {'before':>7s} {'after':>7s}")
    for k in sorted(set(res["before"][0]) | set(res["after"][0])):
        print(f"{k:58s} {res['before'][0][k]:7d} {res['after'][0][k]:7d}")
    print(f"{'records with >=1 listed issue':58s} {res['before'][1]:7d} {res['after'][1]:7d}")
    audit = load(before.replace(".jsonl", ".audit.jsonl"))
    for name in ("before", "after"):
        leaks, checked = still_present(audit, rec_by, res[name][3])
        print(f"identifier surfaces removed by the before run: {checked}; still present in the {name} output: {dict(leaks) or 0}")
    for name in ("before", "after"):
        m = res[name][2]
        r = m["run"]
        print(f"{name}: code {m['code_sha256'][:12]} gpu={m.get('gpu_used')} decisions={r['decisions']} "
              f"s/rec={r['seconds_per_record']} load={r['model_load_seconds']}s peakRAM={r.get('peak_memory_mb_incl_workers')}MB "
              f"escalation={r['escalation_levels']} below_k={r['qi_verification']['records_below_k']}")


if __name__ == "__main__":
    main()
