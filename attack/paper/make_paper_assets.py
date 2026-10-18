"""Tables (JSON) and figures for the evaluation report, from the attack and audit outputs.

    .venv/bin/python attack/paper/make_paper_assets.py --runs attack/results/eval_paper --audits outputs/eval_paper

--runs   folder with one sub-folder per policy (summary.json, per_record.jsonl, per_record_original.jsonl)
--audits folder with eval.<policy>.audit.json (attack.audit_masks) and eval.<policy>.manifest.json (MedAnon)
Writes attack/paper/tables.json and attack/paper/figures/*.png. Aggregates only: no record text.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attack.paper.mask_review import label_of  # noqa: E402

HERE = Path(__file__).parent
POLICIES = ("default", "age_bands", "age_bands_strict_dates")
LABEL = {"original": "original (no anonymization)", "default": "MedAnon defaults", "age_bands": "MedAnon age_bands",
         "age_bands_strict_dates": "MedAnon age_bands + strict dates"}
# Reference categorical palette (dataviz skill), fixed order; text in ink tokens, never series colors.
COLOR = {"original": "#eb6834", "default": "#2a78d6", "age_bands": "#1baf7a", "age_bands_strict_dates": "#6250d6"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
N_POP = 335e6


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion k/n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def recovery_breakdown(anon_rows: list[dict]) -> dict:
    """Correct guesses split by what the placeholder hid: a real identifier, or content MedAnon masked by
    mistake (mask_review.py labels). Recovering an over-redacted 'Lachman' is not a privacy breach."""
    out = {"identifier": [], "overredacted_content": []}
    scorable = {"identifier": 0, "overredacted_content": 0}
    for r in anon_rows:
        for g in r.get("recovery") or []:
            if not g["scorable"]:
                continue
            fp = any(label_of(r["encounter_id"], t) for t in g["truth"])
            key = "overredacted_content" if fp else "identifier"
            scorable[key] += 1
            if g["correct"]:  # a correct name guess IS the patient's name: never written out
                out[key].append({"encounter_id": r["encounter_id"], "placeholder": g["placeholder"], "type": g["type"],
                                 "guess": "<name redacted>" if g["type"] == "name" else g["guess"],
                                 "confidence": g["confidence"]})
    res = {}
    for key, hits in out.items():
        n = scorable[key]
        res[key] = {"n_scorable": n, "n_correct": len(hits), "rate": len(hits) / n if n else None,
                    "ci95": wilson(len(hits), n), "examples": hits}
    non_name = [g for r in anon_rows for g in r.get("recovery") or [] if g["scorable"] and g["type"] != "name"
                and not any(label_of(r["encounter_id"], t) for t in g["truth"])]
    k = sum(g["correct"] for g in non_name)
    res["identifier_excluding_names"] = {"n_scorable": len(non_name), "n_correct": k,
                                         "rate": k / len(non_name) if non_name else None, "ci95": wilson(k, len(non_name))}
    return res


def variant_metrics(records: list[dict], keep) -> dict:
    """Re-aggregate counted attributes under a filter (sensitivity analysis)."""
    bits = []
    for r in records:
        b = sum(a["bits"] for a in r["attributes"] if a["counted"] and keep(a))
        bits.append(b)
    lam = [max((N_POP - 1) * 2.0 ** -b, 1e-12) for b in bits]
    risk = [-math.expm1(-x) / x for x in lam]
    return {"mean_bits": statistics.mean(bits), "median_bits": statistics.median(bits), "mean_risk": statistics.mean(risk),
            "pct_likely_unique": 100 * sum(x < 1 for x in lam) / len(lam)}


VARIANTS = {
    "full": lambda a: True,
    "without_llm_rarity_buckets": lambda a: not (a["category"] == "other_facts" and a["p_source"] == "bucket"),
    "core_quasi_identifiers_only": lambda a: a["category"] != "other_facts",
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def ecdf_figure(bits: dict[str, list[float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    for key, xs in bits.items():
        xs = sorted(xs)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", color=COLOR[key], linewidth=2, label=LABEL[key])
    ax.axvline(math.log2(N_POP), color=INK_2, linewidth=1, linestyle="--")
    ax.text(math.log2(N_POP) - 0.3, 0.5, "unique in the US\n(≈28.3 bits)", color=INK_2, fontsize=8, ha="right")
    style(ax, "Identifying information per record", "bits (sum of −log2 p over counted attributes)")
    ax.set_ylabel("share of records ≤ x", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def category_figure(per_cat: dict[str, dict[str, float]], out: Path) -> None:
    keys = list(per_cat)
    cats = sorted({c for v in per_cat.values() for c, b in v.items() if b > 0.05},
                  key=lambda c: max(v.get(c, 0) for v in per_cat.values()))
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(cats) * len(keys) + 1.6), facecolor=SURFACE)
    h = 0.8 / len(keys)
    for k_i, key in enumerate(reversed(keys)):
        ys = [i + (k_i - (len(keys) - 1) / 2) * h for i in range(len(cats))]
        ax.barh(ys, [per_cat[key].get(c, 0) for c in cats], h * 0.92, color=COLOR[key], label=LABEL[key],
                edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats], color=INK, fontsize=9)
    style(ax, "Where the identifying bits come from", "mean bits per record")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def overredaction_figure(kinds: dict[str, int], out: Path) -> None:
    items = sorted(kinds.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(items) + 1.4), facecolor=SURFACE)
    bars = ax.barh([k.replace("/", " / ") for k, _ in items], [v for _, v in items], 0.6, color=COLOR["default"])
    for b, (_, v) in zip(bars, items, strict=True):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=8, color=INK_2)
    style(ax, "Clinical content masked as identifiers (67 encounters)", "masked spans")
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--audits", required=True)
    args = ap.parse_args()
    runs, audits = Path(args.runs), Path(args.audits)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)

    tables: dict = {"llm_attack": {}, "audit": {}, "medanon_runs": {}}
    bits, per_cat, per_record = {}, {}, defaultdict(dict)
    for pol in POLICIES:
        d = runs / pol
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        a, o, rec = s["anonymized"], s["original"], s["recovery"]
        anon_rows = [r for r in rows(d / "per_record.jsonl") if not r.get("failed")]
        orig_rows = [r for r in rows(d / "per_record_original.jsonl") if not r.get("failed")]
        n_ok = len(anon_rows)
        tables["llm_attack"][pol] = {
            "meta": {k: s["meta"][k] for k in ("model", "extract_prompt", "recover_prompt", "elapsed_s", "llm")},
            "anonymized": {k: a[k] for k in ("n_records", "n_failed", "mean_risk", "mean_bits", "median_bits", "p90_bits",
                                             "pct_likely_unique", "hallucination_rate", "mean_masking_density")},
            "original": {k: o[k] for k in ("n_records", "n_failed", "mean_risk", "mean_bits", "median_bits", "p90_bits",
                                           "pct_likely_unique", "hallucination_rate")},
            "privacy_gain": s["privacy_gain"], "bits_removed": s["bits_removed"], "warnings": s["warnings"],
            "median_risk_anon": statistics.median(r["risk"] for r in anon_rows),
            "max_risk_anon": max(r["risk"] for r in anon_rows),
            "records_risk_ge_1e-3": sum(r["risk"] >= 1e-3 for r in anon_rows),
            "records_lambda_lt_100": sum(r["lambda"] < 100 for r in anon_rows),
            "recovery": {"overall": rec["overall"], "excluding_names": rec["excluding_names"],
                         "overall_ci95": wilson(rec["overall"]["n_correct"], rec["overall"]["n_scorable"]),
                         "by_type": {t: v for t, v in rec["by_type"].items() if v["n_guesses"]},
                         "by_truth_kind": recovery_breakdown(anon_rows)},
            "mean_bits_by_category_anon": {c: v / n_ok for c, v in a["top_leaking_attributes"].items()},
            "mean_bits_by_category_orig": {c: v / max(len(orig_rows), 1) for c, v in o["top_leaking_attributes"].items()},
            "sensitivity": {name: {"anonymized": variant_metrics(anon_rows, f), "original": variant_metrics(orig_rows, f)}
                            for name, f in VARIANTS.items()},
            "p_source_counts_anon": {src: sum(a["counted"] and a["p_source"] == src for r in anon_rows for a in r["attributes"])
                                     for src in ("census", "bls", "table", "bucket", "constant")},
        }
        bits[pol] = [r["bits"] for r in anon_rows]
        per_cat[pol] = tables["llm_attack"][pol]["mean_bits_by_category_anon"]
        if "original" not in bits:
            bits["original"] = [r["bits"] for r in orig_rows]
            per_cat["original"] = tables["llm_attack"][pol]["mean_bits_by_category_orig"]
        for r in anon_rows:
            per_record[r["encounter_id"]][pol] = round(r["bits"], 2)
    for pol in POLICIES:
        f = audits / f"eval.{pol}.audit.json"
        if f.exists():
            au = json.loads(f.read_text())
            tables["audit"][pol] = {k: (f"{v['records']}/{v['of']}" if isinstance(v, dict) and "records" in v else v)
                                    for k, v in au.items() if k not in ("original", "anonymized")}
        m = audits / f"eval.{pol}.manifest.json"
        if m.exists():
            mm = json.loads(m.read_text())
            tables["medanon_runs"][pol] = {"code_sha256": mm["code_sha256"][:16], "config_sha256": mm["config_sha256"][:16],
                                           **{k: mm["run"][k] for k in ("records", "seconds_per_record", "workers",
                                                                       "model_load_seconds") if k in mm["run"]}}
    tables["per_record_bits"] = per_record
    for pol in POLICIES:  # human review of false-positive masks
        f = HERE / f"mask_review.{pol}.json"
        if f.exists():
            tables.setdefault("mask_review", {})[pol] = {k: v for k, v in json.loads(f.read_text()).items()
                                                        if k != "record_ids"}
    for name in ("round1", "round1_after_fixes", "heldout"):
        f = HERE / "results" / f"attribute_review_{name}.json"
        if f.exists():
            d = json.loads(f.read_text())
            d["precision_ci95"] = wilson(d["labels"].get("valid", 0), d["n_attributes"])
            tables.setdefault("attribute_review", {})[name] = d
    ho = runs / "heldout15_age_bands" / "summary.json"
    if ho.exists():
        h = json.loads(ho.read_text())
        tables["heldout"] = {"meta": {k: h["meta"][k] for k in ("elapsed_s", "llm")},
                             "anonymized": {k: h["anonymized"][k] for k in ("n_records", "n_failed", "mean_risk", "mean_bits",
                                                                           "median_bits", "pct_likely_unique")},
                             "original": {k: h["original"][k] for k in ("mean_risk", "mean_bits", "median_bits")},
                             "privacy_gain": h["privacy_gain"], "bits_removed": h["bits_removed"],
                             "recovery": h["recovery"]["overall"]}
    if "mask_review" in tables:
        overredaction_figure(tables["mask_review"]["age_bands"]["clinical_fp_blocks_by_kind"],
                             HERE / "figures" / "fig3_overredaction.png")
    order = ["original", *[p for p in POLICIES if p in bits]]
    (HERE / "tables.json").write_text(json.dumps(tables, indent=1) + "\n")
    ecdf_figure({k: bits[k] for k in order}, HERE / "figures" / "fig1_bits_ecdf.png")
    category_figure({k: per_cat[k] for k in order}, HERE / "figures" / "fig2_bits_by_category.png")
    print(json.dumps({p: {k: tables["llm_attack"][p][k] for k in ("privacy_gain", "bits_removed", "median_risk_anon",
                                                                    "records_risk_ge_1e-3")} for p in tables["llm_attack"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
