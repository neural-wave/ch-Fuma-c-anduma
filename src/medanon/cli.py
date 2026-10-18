"""Command line interface: ``medanon {anonymize, evaluate, bench, canary, demo}``."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TLDEXTRACT_PUBLIC_SUFFIX_LIST_URLS", "")  # inference never touches the network


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _config(args) -> dict:
    from .config import load_config

    over: dict = {}
    if getattr(args, "no_icd", False):
        over["use_icd_context"] = False
    if getattr(args, "no_k_anonymity", False):
        over["k_anonymity"] = {"enabled": False}
    if getattr(args, "ml", None):
        kind, _, model = args.ml.partition(":")
        over["ml"] = {"enabled": True, "kind": kind, "model": model}
    if getattr(args, "no_ml", False):
        over["ml"] = {"enabled": False}
        over["extra_detectors"] = []
    if getattr(args, "offline", False):
        over.setdefault("ml", {})["allow_download"] = False
    if getattr(args, "device", None):
        over.setdefault("ml", {})["device"] = args.device
    cfg = load_config(getattr(args, "config", None), over)
    for ec in cfg.get("extra_detectors") or []:
        if getattr(args, "offline", False):
            ec["allow_download"] = False
        if getattr(args, "device", None):
            ec["device"] = args.device
    return cfg


# ----------------------------------------------------------------------------------- anonymize
_WORKER = None


def _init_worker(cfg: dict) -> None:
    global _WORKER
    from .pipeline import Anonymizer

    _WORKER = Anonymizer(cfg)


def _detect(record: dict):
    texts, spans, registry, kept = _WORKER.detect(record)
    registry.lex = None  # do not ship the lexicon back to the parent process
    return texts, spans, registry, kept


def cmd_anonymize(args) -> None:
    from .manifest import PeakMemory, build_manifest, file_facts
    from .pipeline import Anonymizer

    cfg = _config(args)
    audit = args.audit or str(Path(args.output).with_suffix("")) + ".audit.jsonl"
    with PeakMemory() as mem:
        t_start = time.perf_counter()
        anon = Anonymizer(cfg)
        t_loaded = time.perf_counter()
        if not cfg["k_anonymity"].get("enabled", True) and args.workers <= 1:
            # streaming mode: constant memory, one record in flight (no dataset-level k-anonymity step)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            n = n_dec = 0
            with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8", newline="\n") as fout, \
                    open(audit, "w", encoding="utf-8", newline="\n") as faud:
                for line in fin:
                    if not line.strip():
                        continue
                    out, decs = anon.anonymize(json.loads(line), include_text=args.audit_include_text)
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    for d in decs:
                        faud.write(json.dumps(d, ensure_ascii=False) + "\n")
                    n, n_dec = n + 1, n_dec + len(decs)
            stats = {"mode": "streaming (no k-anonymity step)"}
        else:
            records = _read_jsonl(args.input)
            if args.workers > 1:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(cfg,)) as ex:
                    detections = list(ex.map(_detect, records, chunksize=4))
            else:
                detections = [anon.detect(r) for r in records]
            results, stats = anon.anonymize_dataset(records, include_text=args.audit_include_text, detections=detections)
            _write_jsonl(args.output, (r for r, _ in results))
            _write_jsonl(audit, (d for _, decs in results for d in decs))
            n, n_dec = len(records), sum(len(d) for _, d in results)
        t_done = time.perf_counter()
    report = {
        "records": n, "decisions": n_dec, "audit_contains_original_text": bool(args.audit_include_text),
        "model_load_seconds": round(t_loaded - t_start, 1), "processing_seconds": round(t_done - t_loaded, 1),
        "seconds_per_record": round((t_done - t_loaded) / max(1, n), 3), "workers": args.workers,
        "peak_memory_mb_incl_workers": mem.peak_mb, **stats,
    }
    manifest = build_manifest(anon, run=report, input=file_facts(args.input), output=file_facts(args.output),
                              audit=file_facts(audit))
    manifest_path = str(Path(args.output).with_suffix("")) + ".manifest.json"
    Path(manifest_path).write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps({**report, "output": args.output, "audit": audit, "manifest": manifest_path,
                      "ml_detector": manifest["detectors"]["ml"]["status"]}, indent=1))


def cmd_validate(args) -> None:
    """Strict deliverable check: count, order, unique ids, keys, types, untouched non-text fields."""
    from .evaluation.metrics import validate_deliverable

    res = validate_deliverable(args.original, args.anonymized)
    print(json.dumps(res, indent=1))
    if not res["ok"]:
        sys.exit(1)


# ----------------------------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> None:
    from .evaluation import metrics as M

    orig = _read_jsonl(args.original)
    anon = _read_jsonl(args.anonymized)
    report = {"integrity": M.integrity(orig, anon),
              "utility": {"concepts": M.concept_recall(orig, anon), "icd_terms": M.icd_term_retention(orig, anon)},
              "privacy": {"name_seeds": M.name_seed_leakage(orig, anon)},
              "edits": M.edit_stats(orig, anon)}
    if args.train:
        train = _read_jsonl(args.train)
        train_anon = _read_jsonl(args.train_anonymized) if args.train_anonymized else None
        report["utility"]["icd_proxy"] = M.icd_proxy_f1(train, orig, anon, train_anon)
    if not args.skip_reference:
        report["privacy"]["reference_detector"] = M.leakage_audit(orig, anon)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_bench(args) -> None:
    from .evaluation.bench import run_synthetic
    from .pipeline import Anonymizer

    print(json.dumps(run_synthetic(Anonymizer(_config(args)), n=args.n, show_errors=args.errors), indent=1))


def cmd_canary(args) -> None:
    from .evaluation.canary import evaluate
    from .pipeline import Anonymizer

    res = evaluate(_read_jsonl(args.input)[: args.limit or None], Anonymizer(_config(args)), rounds=args.rounds)
    print(json.dumps(res, indent=1))


def cmd_prepare_models(args) -> None:
    """Download optional ML weights into the local Hugging Face cache (the only networked step)."""
    from .config import DEFAULTS
    from .ner import MLDetector

    specs = args.ml or [f"{DEFAULTS['ml']['kind']}:{DEFAULTS['ml']['model']}"] + \
        [f"{ec['kind']}:{ec['model']}" for ec in DEFAULTS.get("extra_detectors") or [] if ec.get("kind") == "hf"]
    for spec in specs:
        kind, _, model = spec.partition(":")
        MLDetector(kind, model, allow_download=True)
        print(f"cached {kind}:{model}")


def cmd_demo(args) -> None:
    from .evaluation.synthetic import generate
    from .pipeline import Anonymizer

    rec = {k: v for k, v in generate(1, seed=args.seed)[0].items() if not k.startswith("_")}
    out, decs = Anonymizer(_config(args)).anonymize(rec, include_text=True)
    for fld in ("dialogue", "note"):
        print(f"===== {fld} (synthetic) - original\n{rec[fld]}\n===== {fld} - anonymised\n{out[fld]}\n")
    print("===== explanations")
    for d in decs:
        print(f"- [{d['field']}] {d.get('original', d.get('text'))!r} -> {d['replacement']!r}: {d['explanation'][:160]}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="medanon", description="Context-aware anonymisation of doctor-patient encounters")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", help="YAML config (defaults: medanon.config.DEFAULTS)")
        p.add_argument("--no-icd", action="store_true", help="do not use ICD-10 descriptions as context (bonus task)")
        p.add_argument("--ml", help="ML detector kind:model (default hf:dslim/bert-base-NER-uncased, cascade mode)")
        p.add_argument("--no-ml", action="store_true", help="fast profile: rules + spaCy only (no torch needed)")
        p.add_argument("--offline", action="store_true", help="forbid any model download (fails if weights are missing)")
        p.add_argument("--device", choices=["cpu", "cuda", "auto"],
                       help="device of the transformer detectors (default cpu; auto = CUDA when available)")

    p = sub.add_parser("validate", help="strict structural check of an anonymised deliverable")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("anonymize", help="anonymise a JSONL file")
    common(p)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--audit", help="explanation log (default: <output>.audit.jsonl)")
    p.add_argument("--audit-include-text", action="store_true", help="store original surfaces in the audit (sensitive!)")
    p.add_argument("--no-k-anonymity", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_anonymize)

    p = sub.add_parser("evaluate", help="local privacy/utility proxies (not official scores)")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.add_argument("--train", help="original dev split for the ICD-10 proxy coder")
    p.add_argument("--train-anonymized", help="anonymised dev split (train-on-anonymised scenario)")
    p.add_argument("--report")
    p.add_argument("--skip-reference", action="store_true", help="skip the independent Presidio leakage audit")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("bench", help="synthetic gold benchmark")
    common(p)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--errors", type=int, default=0)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("canary", help="canary-injection recall on real records")
    common(p)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rounds", type=int, default=1)
    p.set_defaults(func=cmd_canary)

    p = sub.add_parser("prepare-models", help="download optional ML detector weights (setup time only)")
    p.add_argument("--ml", action="append", help="kind:model, e.g. gliner:knowledgator/gliner-pii-base-v1.0")
    p.set_defaults(func=cmd_prepare_models)

    p = sub.add_parser("demo", help="show a synthetic encounter before/after with explanations")
    common(p)
    p.add_argument("--seed", type=int, default=3)
    p.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
