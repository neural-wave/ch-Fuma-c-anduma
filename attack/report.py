"""Step 7: per_record.jsonl, summary.json, report.md and the leaking-attributes chart."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-2, surface and ink tokens.
BLUE, ORANGE, SURFACE, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n")


def plot_leaks(path: Path, anon: dict[str, float], orig: dict[str, float] | None) -> None:
    """anon/orig: mean bits per record for each attribute category."""
    cats = sorted((c for c in set(anon) | set(orig or {}) if anon.get(c, 0) > 0 or (orig or {}).get(c, 0) > 0),
                  key=lambda c: (anon.get(c, 0), (orig or {}).get(c, 0)))
    if not cats:
        cats = ["(none)"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cats) * (2 if orig else 1) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(cats))
    h = 0.38 if orig else 0.6
    anon_vals = [anon.get(c, 0) for c in cats]
    if orig:
        ax.barh([i - h / 2 - 0.01 for i in y], [orig.get(c, 0) for c in cats], h,
                color=ORANGE, label="original", edgecolor=SURFACE, linewidth=2)
    bars = ax.barh([i + (h / 2 + 0.01 if orig else 0) for i in y], anon_vals, h, color=BLUE,
                   label="anonymized", edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, anon_vals, strict=True):
        if v > 0:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {v:.1f}", va="center",
                    fontsize=8, color=INK_2)
    ax.set_yticks(list(y), [c.replace("_", " ") for c in cats], color=INK, fontsize=9)
    ax.set_xlabel("bits per record (mean over records)", color=INK_2, fontsize=9)
    ax.set_title("Identifying information left, by attribute", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if orig:
        handles, labels = ax.get_legend_handles_labels()  # anonymized is drawn on top: list it first
        ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v, kind: str = "") -> str:
    if v is None:
        return "–"
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "rate":
        return f"{100 * v:.1f}%"
    if kind == "sci":
        return f"{v:.2e}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _cell(s: str | None, limit: int = 90) -> str:
    s = (s or "").replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def write_report(out: Path, meta: dict, summary: dict, anon_rows: list[dict]) -> None:
    anon, orig = summary["anonymized"], summary.get("original")
    lines = [
        "# Re-identification attack report", "",
        f"- Attacker model: `{meta['model']}` (backend `{meta['backend']}`), prompts "
        f"`{meta['extract_prompt']}` / `{meta['recover_prompt']}`",
        f"- Anonymized input: `{meta['input']}`"
        + (f"; original: `{meta['original']}`" if meta.get("original") else ""),
        f"- Records: {anon['n_records']} ({anon['n_failed']} failed); generated {meta['generated']}", "",
        "## Summary", "",
        "| metric | anonymized |" + (" original |" if orig else ""),
        "|---|---|" + ("---|" if orig else ""),
    ]
    metrics = [("mean_risk (headline, lower = better)", "mean_risk", "sci"), ("mean bits", "mean_bits", ""),
               ("median bits", "median_bits", ""), ("p90 bits", "p90_bits", ""),
               ("likely unique (λ < 1)", "pct_likely_unique", "pct"),
               ("hallucination rate", "hallucination_rate", "rate"),
               ("masking density (placeholders / 1k chars)", "mean_masking_density", "")]
    for label, key, kind in metrics:
        lines.append(f"| {label} | {_fmt(anon.get(key), kind)} |" + (f" {_fmt(orig.get(key), kind)} |" if orig else ""))
    if orig:
        lines += ["", f"**Privacy gain** (1 − risk_anon / risk_orig): {_fmt(summary['privacy_gain'], 'rate')} · "
                      f"**bits removed** per record: {_fmt(summary['bits_removed'])}"]
    for w in summary.get("warnings", []):
        lines += ["", f"> ⚠️ {w}"]

    lines += ["", "## Top leaking attributes", "", "![top leaking attributes](top_leaking_attributes.png)", "",
              "| category | total bits (anonymized) |" + (" total bits (original) |" if orig else ""),
              "|---|---|" + ("---|" if orig else "")]
    for cat, bits in anon["top_leaking_attributes"].items():
        orig_cell = f" {orig['top_leaking_attributes'].get(cat, 0):.1f} |" if orig else ""
        lines.append(f"| {cat} | {bits:.1f} |" + orig_cell)

    lines += ["", "## The 5 riskiest records", ""]
    for r in sorted((r for r in anon_rows if not r.get("failed")), key=lambda r: -r["risk"])[:5]:
        lines += [f"### {r['encounter_id']} — risk {r['risk']:.2e}, λ = {r['lambda']:.3g}, {r['bits']:.1f} bits", "",
                  "| category | value | evidence | p (source) | bits | why it identifies |",
                  "|---|---|---|---|---|---|"]
        for a in sorted((a for a in r["attributes"] if a["counted"] and a["bits"] > 0), key=lambda a: -a["bits"]):
            lines.append(f"| {a['category']} | {_cell(a['value'], 50)} | “{_cell(a['evidence'], 70)}” | "
                         f"{a['p']:.2g} ({a['p_source']}: {_cell(a['note'], 40)}) | {a['bits']:.1f} | "
                         f"{_cell(a['rationale'])} |")
        lines.append("")

    rec = summary.get("recovery")
    if rec:
        lines += ["## Recovery attack", "", "| type | guesses | scorable | correct | recovery rate |",
                  "|---|---|---|---|---|"]
        for t, s in [("overall", rec["overall"]), ("excluding names", rec["excluding_names"]),
                     *rec["by_type"].items()]:
            lines.append(f"| {t} | {s['n_guesses']} | {s['n_scorable']} | {s['n_correct']} | "
                         f"{_fmt(s['recovery_rate'], 'rate')} |")
        lines += ["", "### Best successful recoveries", ""]
        for g in rec["best_recoveries"] or [{"none": True}]:
            if g.get("none"):
                lines.append("None.")
                continue
            lines.append(f"- **{g['encounter_id']}** `{g['placeholder']}` → “{g['guess']}” (truth: "
                         f"“{_cell(', '.join(g['truth']), 60)}”, {g['confidence']} confidence). "
                         f"{_cell(g['reasoning'], 300)}")
        if rec["unverified_high_confidence"]:
            lines += ["", "### High-confidence inferences that cannot be checked against the original", ""]
            for g in rec["unverified_high_confidence"]:
                lines.append(f"- **{g['encounter_id']}** `{g['placeholder']}` ({g['type']}) → “{g['guess']}”. "
                             f"{_cell(g['reasoning'], 300)}")
        lines.append("")

    lines += ["## Caveats", "", "Synthetic patients: this is expected uniqueness under approximate US population "
              "statistics with an independence assumption, not an actual re-identification. Results depend on "
              "the attacker model above. See `attack/README.md` for definitions and limitations.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
