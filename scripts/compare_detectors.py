"""Head-to-head comparison of detector configurations (rules+spaCy baseline vs. added ML NER).

For each configuration (run in a fresh process for clean memory numbers):
  * synthetic gold benchmark  -> entity recall, char precision, medical-trap preservation
  * canary injection into real dev records -> recall in real clinical context
  * real dev records without injection -> how many *extra* edits the model makes and how many of
    those are made only of common/medical words (privacy-safe false-positive proxy)
  * seconds per record and peak working-set memory

Usage:  python scripts/compare_detectors.py --dev data/patient_doctor_encounters.jsonl --out outputs/detector_comparison.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_GLINER = {"kind": "gliner", "model": "knowledgator/gliner-pii-base-v1.0"}
_GLINER2 = {"kind": "gliner2", "model": "fastino/gliner2-privacy-filter-PII-multi"}
_BERT = {"kind": "hf", "model": "dslim/bert-base-NER-uncased"}
CONFIGS = {
    "rules+spacy": {"ml": {"enabled": False}},
    "+gliner-pii-base:full": {"ml": {"enabled": True, **_GLINER, "threshold": 0.5, "mode": "full", "fields": ["dialogue"]}},
    "+gliner2-privacy:full": {"ml": {"enabled": True, **_GLINER2, "threshold": 0.5, "mode": "full", "fields": ["dialogue"]}},
    "+bert-uncased-ner:full": {"ml": {"enabled": True, **_BERT, "threshold": 0.5, "mode": "full", "fields": ["dialogue"]}},
    "+gliner-pii-base:cascade": {"ml": {"enabled": True, **_GLINER, "threshold": 0.5, "mode": "cascade", "fields": ["dialogue"]}},
    "+gliner2-privacy:cascade": {"ml": {"enabled": True, **_GLINER2, "threshold": 0.5, "mode": "cascade", "fields": ["dialogue"]}},
    "+bert-uncased-ner:cascade": {"ml": {"enabled": True, **_BERT, "threshold": 0.5, "mode": "cascade", "fields": ["dialogue"]}},
}


def run_one(name: str, dev: str, n_dev: int) -> dict:
    import psutil

    from medanon import Anonymizer, load_config
    from medanon.evaluation import canary, synthetic
    from medanon.evaluation.diagnostics import safe_surface

    proc = psutil.Process()
    t0 = time.perf_counter()
    anon = Anonymizer(load_config(overrides=CONFIGS[name]))
    load_s = time.perf_counter() - t0
    res = {"config": name, "load_seconds": round(load_s, 1)}

    recs = synthetic.generate(60, seed=21, style="aci") + synthetic.generate(15, seed=22, style="discharge")
    outs, decs = [], []
    t0 = time.perf_counter()
    for r in recs:
        o, d = anon.anonymize({k: v for k, v in r.items() if not k.startswith("_")})
        outs.append(o)
        decs.append(d)
    res["synthetic"] = synthetic.score(recs, outs, decs)
    res["synthetic"]["sec_per_record"] = round((time.perf_counter() - t0) / len(recs), 3)

    with open(dev, encoding="utf-8") as f:
        dev_recs = [json.loads(line) for line in f if line.strip()][:n_dev]
    t0 = time.perf_counter()
    can = canary.evaluate(dev_recs, anon, seed=5, rounds=1)
    res["canary"] = {k: v for k, v in can.items() if "/tpl:" not in k or "going" in k}
    res["canary_sec_per_record"] = round((time.perf_counter() - t0) / len(dev_recs), 3)

    edits = common_only = 0
    for r in dev_recs:
        _, d = anon.anonymize(r)
        for x in d:
            if x["action"] == "keep" or x["label"] == "LAB_VALUE":
                continue
            edits += 1
            surface = r[x["field"]][x["start"]:x["end"]]
            common_only += safe_surface(surface) == surface.lower() and x["label"] in {"PERSON", "LOCATION", "ORGANIZATION", "OCCUPATION", "NATIONALITY"}
    res["dev_edits"] = edits
    res["dev_edits_common_or_medical_words"] = common_only
    mem = proc.memory_info()
    res["peak_memory_mb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e6)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/patient_doctor_encounters.jsonl")
    ap.add_argument("--n-dev", type=int, default=60)
    ap.add_argument("--out", default="outputs/detector_comparison.json")
    ap.add_argument("--only")
    ap.add_argument("--child")
    args = ap.parse_args()
    if args.child:
        print(json.dumps(run_one(args.child, args.dev, args.n_dev)))
        return
    results = []
    for name in CONFIGS:
        if args.only and name not in args.only.split(","):
            continue
        cmd = [sys.executable, __file__, "--child", name, "--dev", args.dev, "--n-dev", str(args.n_dev)]
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        line = [x for x in out.stdout.splitlines() if x.startswith("{")]
        if not line:
            print(name, "FAILED", out.stderr[-2000:])
            continue
        results.append(json.loads(line[-1]))
        r = results[-1]
        print(f"{name:<20} synth_recall={r['synthetic']['recall_coverage']} prec={r['synthetic']['char_precision']} "
              f"traps={r['synthetic']['trap_preservation']} canary={r['canary']['ALL']['recall']} "
              f"canary_person_dialogue={r['canary']['PERSON/dialogue']['recall']} edits={r['dev_edits']} "
              f"common_word_edits={r['dev_edits_common_or_medical_words']} s/rec={r['canary_sec_per_record']} "
              f"peakMB={r['peak_memory_mb']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
