"""Controlled model experiment: baseline vs. baseline + one candidate detector.

Same records, seeds, thresholds and scorer for every configuration; each configuration runs in a
fresh process (clean memory numbers). Evidence sources are kept separate:
  1. held-out adversarial synthetic set (exact gold)            -> P / R / F1 with Wilson CIs
  2. fixed validation subset of real dev encounters (no gold)   -> incremental spans vs. baseline,
     utility integrity checks; surfaces only in a local review file for humans
Usage:
  python scripts/model_experiments.py --configs baseline,A_healthcare_brain,C_openmed,B_qwen_cascade
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

OUT = Path("outputs/experiments")
REVIEW = Path("outputs/review")
VAL_IDS = Path("configs/validation_subset.json")
DEV = "data/patient_doctor_encounters.jsonl"

CONFIGS = {
    "baseline": {"extra_detectors": []},  # historical baseline = no OpenMed (it became part of the default)
    "A_healthcare_brain": {"extra_detectors": [{"kind": "hf", "model": "genzeonplatform/healthcare-brain-ner",
                                                "label_map": "healthcare-brain", "mode": "full", "threshold": 0.5,
                                                "fields": ["dialogue", "note"]}]},
    "C_openmed": {"extra_detectors": [{"kind": "hf", "model": "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1",
                                       "label_map": "openmed-pii", "mode": "full", "threshold": 0.5,
                                       "fields": ["dialogue", "note"]}]},
    "C_openmed_cascade": {"extra_detectors": [{"kind": "hf", "model": "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1",
                                               "label_map": "openmed-pii", "mode": "cascade", "threshold": 0.5,
                                               "fields": ["dialogue", "note"]}]},
    "B_qwen_cascade": {"extra_detectors": [{"kind": "llm", "model": "unsloth/Qwen3.5-2B-GGUF", "mode": "cascade",
                                            "fields": ["dialogue"], "prompt": "few-shot", "threads": 10}]},
    "B_qwen_cascade_zeroshot": {"extra_detectors": [{"kind": "llm", "model": "unsloth/Qwen3.5-2B-GGUF", "mode": "cascade",
                                                     "fields": ["dialogue"], "prompt": "zero-shot", "threads": 10}]},
    "B_qwen_cascade_nooutlet": {"extra_detectors": [{"kind": "llm", "model": "unsloth/Qwen3.5-2B-GGUF", "mode": "cascade",
                                                     "fields": ["dialogue"], "prompt": "few-shot", "threads": 10,
                                                     "medical_outlet": False}]},
}
NEGATION = re.compile(r"\b(?:no|not|denies|denied|deny|without|negative|never|none|n't)\b", re.I)
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validation_ids(n: int = 25, seed: int = 17) -> list[str]:
    """Fixed, stratified (by dataset style) subset of the dev split; ids only are stored."""
    if VAL_IDS.exists():
        return json.loads(VAL_IDS.read_text())["encounter_ids"]
    recs = read(DEV)
    rng = random.Random(seed)
    by = {}
    for r in recs:
        by.setdefault(r["dataset"], []).append(r["encounter_id"])
    ids = []
    for ds, lst in sorted(by.items()):
        k = round(n * len(lst) / len(recs))
        ids += rng.sample(sorted(lst), k)
    VAL_IDS.write_text(json.dumps({"seed": seed, "source": DEV, "stratified_by": "dataset", "encounter_ids": sorted(ids)},
                                  indent=1))
    return sorted(ids)


def utility_checks(orig: dict, out: dict, decs: list[dict]) -> Counter:
    c = Counter()
    for f in ("dialogue", "note"):
        c["negation_cues_orig"] += len(NEGATION.findall(orig[f]))
        c["negation_cues_out"] += len(NEGATION.findall(out[f]))
        masked = list(orig[f])
        for d in decs:
            if d["field"] == f and d["action"] != "keep":
                for i in range(d["start"], d["end"]):
                    masked[i] = " "
        outside = Counter(NUM.findall("".join(masked)))
        in_out = Counter(NUM.findall(re.sub(r"\[[^\]]*\]", " ", out[f])))
        c["numbers_outside_spans"] += sum(outside.values())
        c["numbers_outside_spans_missing"] += sum((outside - in_out).values())
    return c


def run_child(name: str, n_adv: int, n_val: int = 0) -> dict:
    from medanon import Anonymizer, load_config
    from medanon.evaluation import adversarial, scoring
    from medanon.manifest import PeakMemory

    res = {"config": name, "overrides": CONFIGS[name]}
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        anon = Anonymizer(load_config(overrides=CONFIGS[name]))
        res["cold_load_seconds"] = round(time.perf_counter() - t0, 1)
        # 1) held-out adversarial synthetic set
        recs = adversarial.generate(n_adv, bank="test")
        t0 = time.perf_counter()
        outs, decs = zip(*(anon.anonymize({k: v for k, v in r.items() if not k.startswith("_")}) for r in recs))
        res["adv_seconds_per_encounter"] = round((time.perf_counter() - t0) / len(recs), 3)
        res["adversarial_test"] = scoring.score(recs, list(outs), list(decs))
        # 2) real validation subset (no gold): spans + utility integrity
        ids = set(validation_ids())
        val = [r for r in read(DEV) if r["encounter_id"] in ids][: n_val or None]
        t0 = time.perf_counter()
        spans, util = {}, Counter()
        for r in val:
            out, d = anon.anonymize(r)
            spans[r["encounter_id"]] = [(x["field"], x["start"], x["end"], x["label"], x["action"], ",".join(x["detectors"]))
                                        for x in d]
            util += utility_checks(r, out, d)
        res["val_seconds_per_encounter"] = round((time.perf_counter() - t0) / len(val), 3)
        res["val_records"] = len(val)
        res["val_utility"] = dict(util)
        for _, det in anon.extra:
            if hasattr(det, "stats"):
                res["llm_stats"] = dict(det.stats)
    res["peak_memory_mb"] = mem.peak_mb
    (OUT / f"{name}.spans.json").write_text(json.dumps(spans))  # offsets/labels only, no text
    return res


def incremental(name: str, base: dict, cand: dict, records: dict) -> dict:
    """Spans replaced by `cand` but not by `base` (and vice versa) on real records, as privacy-safe
    categories. Surfaces go to a local review file only."""
    from medanon.evaluation.diagnostics import safe_surface
    from medanon.lexicon import load_lexicon

    lex = load_lexicon()
    added, lost = Counter(), Counter()
    review = []
    for eid, spans in cand.items():
        b = [s for s in base.get(eid, []) if s[4] != "keep"]
        for s in spans:
            if s[4] == "keep" or s[3] == "LAB_VALUE":
                continue
            if any(x[0] == s[0] and x[1] < s[2] and s[1] < x[2] for x in b):
                continue
            text = records[eid][s[0]][s[1]:s[2]]
            words = re.findall(r"[^\W\d_][\w'\-]*", text)
            shown = safe_surface(text)
            medical = any(lex.medical_strength(w.lower()) == 2 or w.lower() in lex.eponyms for w in words)
            cls = "medical-word" if medical else "common-word" if shown == text.lower() else \
                "name-lexicon" if any(lex.in_name_lexicon(w.lower()) for w in words) else "rare/other"
            added[(s[3], s[0], cls)] += 1
            review.append({"encounter_id": eid, "field": s[0], "label": s[3], "detectors": s[5], "text": text,
                           "context": records[eid][s[0]][max(0, s[1] - 80):s[2] + 80]})
        c = [s for s in spans if s[4] != "keep"]
        for s in base.get(eid, []):
            if s[4] != "keep" and s[3] != "LAB_VALUE" and not any(x[0] == s[0] and x[1] < s[2] and s[1] < x[2] for x in c):
                lost[(s[3], s[0])] += 1
    REVIEW.mkdir(parents=True, exist_ok=True)
    with open(REVIEW / f"{name}_added_spans.jsonl", "w", encoding="utf-8") as f:  # local, git-ignored
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"added_total": sum(added.values()), "added_by": {"|".join(k): v for k, v in added.most_common()},
            "baseline_spans_lost": {"|".join(k): v for k, v in lost.most_common()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline,A_healthcare_brain,C_openmed,B_qwen_cascade")
    ap.add_argument("--n-adv", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=0, help="limit validation records (expensive configs)")
    ap.add_argument("--child")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        print("RESULT " + json.dumps(run_child(args.child, args.n_adv, args.n_val)))
        return
    validation_ids()
    results = {}
    for name in args.configs.split(","):
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--child", name, "--n-adv", str(args.n_adv), "--n-val", str(args.n_val)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(f"{name}: FAILED\n{p.stderr[-3000:]}", flush=True)
            continue
        results[name] = json.loads(line[-1][7:])
        a = results[name]["adversarial_test"]
        print(f"{name:<26} adv R={a['recall']['value']} {a['recall']['ci95']} P={a['precision']['value']} "
              f"F1={a['f1']} traps={a['trap_preservation_all']['value']} | s/enc adv={results[name]['adv_seconds_per_encounter']} "
              f"val={results[name]['val_seconds_per_encounter']} | load={results[name]['cold_load_seconds']}s "
              f"peakMB={results[name]['peak_memory_mb']} | wall={time.perf_counter() - t0:.0f}s", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(results[name], indent=1))
    if "baseline" in results or (OUT / "baseline.spans.json").exists():
        base = json.loads((OUT / "baseline.spans.json").read_text())
        records = {r["encounter_id"]: r for r in read(DEV)}
        for name in results:
            if name == "baseline":
                continue
            cand = json.loads((OUT / f"{name}.spans.json").read_text())
            results[name]["real_incremental"] = incremental(name, base, cand, records)
            (OUT / f"{name}.json").write_text(json.dumps(results[name], indent=1))
            print(f"{name}: real-data incremental", json.dumps(results[name]["real_incremental"])[:900], flush=True)


if __name__ == "__main__":
    main()
