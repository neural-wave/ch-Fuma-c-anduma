"""Reproduce every number reported in README/REPORT (aggregate metrics only, no text).

    python scripts/run_experiments.py --dev data/patient_doctor_encounters.jsonl \
        --eval data/eval_patient_doctor_encounters.jsonl --out outputs

Steps: anonymise dev + eval (deliverable: outputs/eval_patient_doctor_encounters.anonymized.jsonl),
integrity checks, utility proxies (concept recall, ICD-term retention, ICD-10 coder F1), privacy
proxies (independent Presidio audit, name-seed leakage), adversarial attacks, canary recall on dev,
synthetic gold benchmark, the no-ICD-context ablation, and runtime / memory.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    import psutil

    from medanon import Anonymizer, load_config
    from medanon.evaluation import attacks, canary, metrics, synthetic

    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/patient_doctor_encounters.jsonl")
    ap.add_argument("--eval", default="data/eval_patient_doctor_encounters.jsonl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config")
    ap.add_argument("--skip-canary", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    t0 = time.perf_counter()
    anon = Anonymizer(load_config(args.config))
    res["load_seconds"] = round(time.perf_counter() - t0, 1)

    data = {"dev": read(args.dev), "eval": read(args.eval)}
    released, dets = {}, {}
    for split, recs in data.items():
        t0 = time.perf_counter()
        dets[split] = [anon.detect(r) for r in recs]
        results, stats = anon.anonymize_dataset(recs, detections=dets[split])
        secs = time.perf_counter() - t0
        released[split] = [r for r, _ in results]
        name = Path(args.dev if split == "dev" else args.eval).stem
        write(out / f"{name}.anonymized.jsonl", released[split])
        write(out / f"{name}.audit.jsonl", (d for _, decs in results for d in decs))
        decs = [d for _, ds in results for d in ds]
        res[f"{split}_run"] = {"records": len(recs), "seconds_per_record": round(secs / len(recs), 3),
                               "decisions": len(decs),
                               "explained": round(sum(bool(d.get("explanation")) for d in decs) / max(1, len(decs)), 4),
                               **stats}
        print(f"[{split}] anonymised {len(recs)} records in {secs:.1f}s", flush=True)

    ev, dv = data["eval"], data["dev"]
    res["integrity"] = {s: metrics.integrity(data[s], released[s]) for s in data}
    res["utility"] = {
        "concepts_eval": metrics.concept_recall(ev, released["eval"]),
        "icd_terms_eval": metrics.icd_term_retention(ev, released["eval"]),
        "icd_proxy": metrics.icd_proxy_f1(dv, ev, released["eval"], released["dev"]),
    }
    print("[utility]", json.dumps(res["utility"]["icd_proxy"]), flush=True)
    ref = metrics.ReferenceDetector()
    res["privacy"] = {s: {"reference_detector": metrics.leakage_audit(data[s], released[s], ref),
                          "name_seeds": metrics.name_seed_leakage(data[s], released[s])} for s in data}
    res["edits"] = {s: metrics.edit_stats(data[s], released[s]) for s in data}
    res["attacks"] = {s: {"linkage": attacks.linkage_attack(anon, data[s], released[s], dets[s]),
                          "lab_linkage": attacks.lab_linkage_attack(data[s], released[s]),
                          "residual_names": attacks.residual_name_attack(data[s], released[s]),
                          "qi_uniqueness": attacks.qi_uniqueness(data[s], released[s]),
                          "perturbation_inversion": attacks.perturbation_inversion_attack(data[s])} for s in data}
    res["deliverable_validation"] = metrics.validate_deliverable(args.eval, str(out / f"{Path(args.eval).stem}.anonymized.jsonl"))
    print("[privacy+attacks] done", flush=True)

    # held-out adversarial synthetic set (exact gold, Wilson CIs) + independent utility integrity
    from medanon.evaluation import adversarial, scoring

    adv = adversarial.generate(40, bank="test")
    a_out, a_dec = zip(*(anon.anonymize({k: v for k, v in r.items() if not k.startswith("_")}) for r in adv))
    res["adversarial_test"] = scoring.score(adv, list(a_out), list(a_dec))
    neg = re.compile(r"\b(?:no|not|denies|denied|deny|without|negative|never|none|n't)\b", re.I)
    res["utility"]["negation_cues"] = {s: {"original": sum(len(neg.findall(r[f])) for r in data[s] for f in ("dialogue", "note")),
                                           "anonymised": sum(len(neg.findall(r[f])) for r in released[s] for f in ("dialogue", "note"))}
                                       for s in data}

    # ablation: no ICD-10 context at inference (bonus task)
    # same perturbation key as the main run, so only the ICD context differs between the two outputs
    no_icd = Anonymizer(load_config(args.config, {"use_icd_context": False, "perturbation_secret": anon.transformer.secret}))
    rel_no_icd = [r for r, _ in no_icd.anonymize_dataset(ev)[0]]
    res["no_icd_ablation"] = {"concepts_eval": metrics.concept_recall(ev, rel_no_icd),
                              "icd_terms_eval": metrics.icd_term_retention(ev, rel_no_icd),
                              "name_seeds": metrics.name_seed_leakage(ev, rel_no_icd),
                              "identical_to_main_output": sum(a == b for a, b in zip(rel_no_icd, released["eval"]))}

    recs = synthetic.generate(120, seed=7) + synthetic.generate(30, seed=8, style="discharge")
    outs, decs = zip(*(anon.anonymize({k: v for k, v in r.items() if not k.startswith("_")}) for r in recs))
    res["synthetic"] = {"aci": synthetic.score(recs[:120], list(outs[:120]), list(decs[:120])),
                        "discharge": synthetic.score(recs[120:], list(outs[120:]), list(decs[120:]))}
    if not args.skip_canary:
        res["canary_dev"] = canary.evaluate(dv, anon, seed=11, rounds=2)
    mem = psutil.Process().memory_info()
    res["peak_memory_mb_whole_suite"] = round(getattr(mem, "peak_wset", mem.rss) / 1e6)
    from medanon.manifest import build_manifest

    res["manifest"] = build_manifest(anon)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("eval_run", "utility", "privacy", "attacks", "adversarial_test")},
                     indent=1, default=str)[:8000])


if __name__ == "__main__":
    main()
