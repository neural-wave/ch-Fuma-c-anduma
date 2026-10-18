"""CLI entry point: python -m attack.run --input anon.jsonl [--original orig.jsonl] [...]"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from . import extract as extract_mod
from . import recover as recover_mod
from .data import PLACEHOLDER_RE, load_jsonl, to_record
from .extract import extract
from .llm import LLMClient, LLMError, run_pool
from .recover import align_placeholders, recover, score_guesses, summarize_recovery
from .report import plot_leaks, write_jsonl, write_report
from .score import compare, score_record, summarize
from .verify import fallback_age, verify

log = logging.getLogger("attack")
DEFAULT_INPUT = "data/output_eval_patient_doctor_encounters.jsonl"
DEFAULT_ORIGINAL = "data/eval_patient_doctor_encounters.jsonl"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m attack.run",
                                 description="Re-identification attack on anonymized encounters.")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="anonymized JSONL")
    ap.add_argument("--original", nargs="?", const=DEFAULT_ORIGINAL, default=None,
                    help="original JSONL: enables the comparison and the recovery attack "
                         f"(bare flag: {DEFAULT_ORIGINAL})")
    ap.add_argument("--model", help="default: $ATTACK_MODEL or qwen3.5:35b")
    ap.add_argument("--base-url", help="default: $ATTACK_BASE_URL or http://localhost:11434/v1")
    ap.add_argument("--backend", choices=["auto", "ollama", "openai"], default="auto")
    ap.add_argument("--num-ctx", type=int, default=16384, help="context window for the Ollama backend")
    ap.add_argument("--out", help="output directory (default: attack/results/<model>_<input>[_limitN])")
    ap.add_argument("--limit", type=int, help="only the first N records")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", default="attack/cache")
    return ap.parse_args(argv)


def analyze(rec: dict, extraction: dict | None, also: dict | None = None, also_label: str = "",
            fuzzy_also: bool = True) -> dict:
    """`also`: the other version's extraction (anonymized <-> original). Each record is scored on its
    own extraction plus the other side's facts whose evidence verifies in its own text, so the comparison
    measures what the anonymizer removed, not which facts one LLM call happened to miss. Facts from the
    original must be quoted verbatim (fuzzy_also=False): a quote holding a masked value can't match."""
    base = {k: rec[k] for k in ("encounter_id", "masking_density", "n_placeholders", "n_icd_codes")}
    if extraction is None:
        return {**base, "failed": True}
    kept, dropped, n_extracted = verify(extraction, rec["text"], rec["icd_descriptions"],
                                        icd_external=rec["icd_external"])
    if not any(a["field"] in ("age", "birth_year") for a in kept) and (age := fallback_age(rec["text"])):
        kept.append(age)
    if also:
        seen = {(a["field"], a["value"].lower()) for a in kept}
        kept += [{**a, also_label: True}
                 for a in verify(also, rec["text"], rec["icd_descriptions"], fuzzy=fuzzy_also,
                                 icd_external=rec["icd_external"])[0]
                 if (a["field"], a["value"].lower()) not in seen]
    return {"encounter_id": rec["encounter_id"], "failed": False, **score_record(kept), **base,
            "n_extracted": n_extracted, "dropped": dropped}


def mean_bits_by_category(summary: dict) -> dict[str, float]:
    n = summary["n_records"] - summary["n_failed"]
    return {k: v / n for k, v in summary["top_leaking_attributes"].items()} if n else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    client = LLMClient(args.base_url, args.model, args.backend, args.num_ctx, args.cache_dir)
    try:
        client.connect()
    except LLMError as e:
        log.error("%s", e)
        return 2

    anon = [to_record(r) for r in load_jsonl(args.input, args.limit)]
    orig: dict[str, dict] = {}
    if args.original:
        wanted = {r["encounter_id"] for r in anon}
        orig = {r["encounter_id"]: to_record(r) for r in load_jsonl(args.original) if r["encounter_id"] in wanted}
        if missing := sorted(wanted - orig.keys()):
            log.warning("%d records have no original: %s", len(missing), ", ".join(missing[:10]))
    paired = [r for r in anon if r["encounter_id"] in orig]

    tasks = ([(extract, r) for r in anon] + [(extract, orig[r["encounter_id"]]) for r in paired]
             + [(recover, r) for r in paired])
    started = time.time()
    results = run_pool(lambda t: t[0](client, t[1]["text"]), tasks, args.workers, "llm calls")
    n_anon, n_paired = len(anon), len(paired)
    anon_ext = results[:n_anon]
    orig_ext = results[n_anon:n_anon + n_paired]
    recoveries = results[n_anon + n_paired:]

    orig_ext_by_id = {r["encounter_id"]: e for r, e in zip(paired, orig_ext, strict=True)}
    anon_rows = [analyze(r, e, orig_ext_by_id.get(r["encounter_id"]), "from_original", fuzzy_also=False)
                 for r, e in zip(anon, anon_ext, strict=True)]
    summary: dict = {"anonymized": summarize(anon_rows)}
    orig_rows: list[dict] = []
    warnings: list[str] = []
    if paired:
        anon_ext_by_id = {r["encounter_id"]: e for r, e in zip(anon, anon_ext, strict=True)}
        orig_rows = [analyze(orig[r["encounter_id"]], e, anon_ext_by_id[r["encounter_id"]], "from_anonymized")
                     for r, e in zip(paired, orig_ext, strict=True)]
        summary["original"] = summarize(orig_rows)
        anon_by_id = {r["encounter_id"]: r for r in anon_rows}
        both = [(anon_by_id[o["encounter_id"]], o) for o in orig_rows
                if not o["failed"] and not anon_by_id[o["encounter_id"]]["failed"]]
        if both:  # the comparison only uses records where both extractions succeeded
            summary.update(compare(summarize([a for a, _ in both]), summarize([o for _, o in both])))
            if summary["privacy_gain"] is not None and summary["privacy_gain"] <= 0:
                warnings.append("risk on the original is not higher than on the anonymized text")
            if worse := [a["encounter_id"] for a, o in both if a["risk"] > o["risk"]]:
                warnings.append(f"{len(worse)} records score riskier anonymized than original "
                                f"(extraction noise or added leaks): {', '.join(worse[:10])}")
        for rec, recovery, o in zip(paired, recoveries, orig_rows, strict=True):
            if recovery is not None:
                truth = align_placeholders(rec["text"], orig[rec["encounter_id"]]["text"])
                anon_by_id[rec["encounter_id"]]["recovery"] = score_guesses(
                    recovery, truth, o.get("attributes", []), set(PLACEHOLDER_RE.findall(rec["text"])))
        summary["recovery"] = summarize_recovery(anon_rows)
    summary["warnings"] = warnings
    for w in warnings:
        log.warning("SANITY: %s", w)

    run_name = re.sub(r"[^A-Za-z0-9._-]+", "-", client.model) + "_" + Path(args.input).stem
    out = Path(args.out or f"attack/results/{run_name}{f'_limit{args.limit}' if args.limit else ''}")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": client.model, "backend": client.backend, "input": args.input, "original": args.original,
            "extract_prompt": extract_mod.PROMPT_VERSION, "recover_prompt": recover_mod.PROMPT_VERSION,
            "generated": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.time() - started, 1),
            "llm": client.stats}
    write_jsonl(out / "per_record.jsonl", anon_rows)
    if orig_rows:
        write_jsonl(out / "per_record_original.jsonl", orig_rows)
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    plot_leaks(out / "top_leaking_attributes.png", mean_bits_by_category(summary["anonymized"]),
               mean_bits_by_category(summary["original"]) if orig_rows else None)
    write_report(out, meta, summary, anon_rows)

    a = summary["anonymized"]
    print(f"\n{a['n_records']} records ({a['n_failed']} failed) | mean_risk {a['mean_risk']:.3e} | "
          f"median bits {a['median_bits']:.1f} | likely unique {a['pct_likely_unique']:.1f}% | "
          f"hallucination {100 * a['hallucination_rate']:.1f}%" if a["mean_risk"] is not None else "\nno record scored")
    if summary.get("privacy_gain") is not None:
        print(f"privacy_gain {100 * summary['privacy_gain']:.1f}% | bits_removed {summary['bits_removed']:.1f}")
    if summary.get("recovery"):
        rec = summary["recovery"]
        print(f"recovery_rate {rec['overall']['recovery_rate']} "
              f"(excluding names {rec['excluding_names']['recovery_rate']})")
    print(f"outputs: {out}/  (LLM calls {client.stats['llm_calls']}, cache hits {client.stats['cache_hits']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
