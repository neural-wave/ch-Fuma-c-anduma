"""Gold-standard scoring with uncertainty: entity-level precision / recall / F1 (Wilson 95 % CIs),
breakdowns by label, field and difficulty, severity-weighted residual identifiers and clinical
trap preservation by type."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

SEVERITY = {"PERSON": 3, "ID": 3, "CONTACT": 3, "DOB": 3, "ADDRESS": 3, "DATE": 2, "AGE": 2, "LOCATION": 2,
            "ORGANIZATION": 2, "OCCUPATION": 1}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def _rate(k: int, n: int) -> dict:
    return {"value": round(k / n, 4) if n else None, "k": k, "n": n, "ci95": wilson(k, n)}


def _policy_kept(decisions: list[dict]) -> dict[str, set[int]]:
    """Characters detected but deliberately published by policy (exact age <= 89, calendar years)."""
    out: dict[str, set[int]] = defaultdict(set)
    for d in decisions:
        if d["action"] == "keep" and d.get("reason_code") == "POLICY_KEEP":
            out[d["field"]].update(range(d["start"], d["end"]))
    return out


def _replaced(decisions: list[dict]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for d in decisions:
        if d["action"] != "keep" and d["label"] not in ("LAB_VALUE",):
            out[d["field"]].update(range(d["start"], d["end"]))
    return out


def _lab_trap_ok(orig: str, anon_text: str) -> bool:
    """A lab trap survives if its analyte words remain and its value moved by <= 3 %."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", orig) if not re.fullmatch(r"(?:is|of|at)", w, re.I)]
    from ..lexicon import words_to_number

    num = re.search(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?(?![A-Za-z\d])|\b[a-z]+ point [a-z]+(?: [a-z]+)?", orig)
    if not num:
        return False
    v = words_to_number(num.group(0)) if not num.group(0)[0].isdigit() else float(num.group(0))
    head = re.escape(orig[:num.start()].strip())
    m = re.search(head + r"\s+(\d+(?:\.\d+)?|[a-z]+ point [a-z]+(?: [a-z]+)?)", anon_text)
    if not m or not words:
        return False
    v2 = float(m.group(1)) if m.group(1)[0].isdigit() else words_to_number(m.group(1))
    return v2 is not None and v is not None and abs(v2 - v) <= 0.03 * abs(v) + 1e-9


def score(records: list[dict], outputs: list[dict], decisions: list[list[dict]]) -> dict:
    tot, hit = Counter(), Counter()
    policy_kept, pol_hit = Counter(), Counter()
    pred_n = pred_tp = 0
    traps_n, traps_ok = Counter(), Counter()
    residual, direct_left = [], []
    for rec, out, decs in zip(records, outputs, decisions):
        repl = _replaced(decs)
        gold_chars = defaultdict(set)
        sev_left = 0
        kept = _policy_kept(decs)
        for g in rec["_gold"]:
            gold_chars[g["field"]].update(range(g["start"], g["end"]))
            text = rec[g["field"]]
            alnum = [i for i in range(g["start"], g["end"]) if text[i].isalnum()]
            # raw recall (historical definition): an entity counts only if it was removed; an entity that the
            # policy deliberately publishes (exact age <= 89, calendar year) stays a miss here and is counted
            # separately, so policy-compliant recall = (removed + kept by policy) / all gold entities
            ok = all(i in repl[g["field"]] for i in alnum)
            kept_ok = not ok and bool(alnum) and all(i in repl[g["field"]] or i in kept[g["field"]] for i in alnum)
            for key in ("ALL", f"label:{g['label']}", f"field:{g['field']}", f"hard:{g['hard']}"):
                tot[key] += 1
                hit[key] += ok
                pol_hit[key] += ok or kept_ok
            if kept_ok:
                policy_kept[g["label"]] += 1
            if not ok:
                sev_left += SEVERITY.get(g["label"], 1)
        residual.append(sev_left)
        direct_left.append(any(SEVERITY.get(g["label"], 1) == 3 and not all(
            i in repl[g["field"]] for i in range(g["start"], g["end"]) if rec[g["field"]][i].isalnum()) for g in rec["_gold"]))
        for d in decs:
            if d["action"] == "keep" or d["label"] == "LAB_VALUE":
                continue
            pred_n += 1
            pred_tp += any(i in gold_chars[d["field"]] for i in range(d["start"], d["end"]))
        for t in rec.get("_traps", []):
            traps_n[t["kind"]] += 1
            if t["kind"] == "lab-value":  # unchanged, or analyte kept and value moved <= 3 %
                ok = t["text"] in out[t["field"]] or _lab_trap_ok(t["text"], out[t["field"]])
            else:
                ok = not any(i in repl[t["field"]] for i in range(t["start"], t["end"])) and t["text"] in out[t["field"]]
            traps_ok[t["kind"]] += ok
    rec_ = hit["ALL"] / tot["ALL"] if tot["ALL"] else 0.0
    prec = pred_tp / pred_n if pred_n else 0.0
    return {
        "records": len(records), "gold_entities": tot["ALL"], "predicted_spans": pred_n,
        "recall": _rate(hit["ALL"], tot["ALL"]), "precision": _rate(pred_tp, pred_n),
        # NOT unconditional privacy recall: counts entities the policy deliberately publishes as compliant
        "policy_compliant_recall": _rate(pol_hit["ALL"], tot["ALL"]),
        "kept_by_policy": dict(policy_kept),
        "f1": round(2 * prec * rec_ / (prec + rec_), 4) if prec + rec_ else 0.0,
        "recall_by": {k: _rate(hit[k], tot[k]) for k in sorted(tot) if k != "ALL"},
        "severity_weighted_residual_per_encounter": round(sum(residual) / max(1, len(residual)), 3),
        "encounters_with_direct_identifier_left": _rate(sum(direct_left), len(direct_left)),
        "trap_preservation": {k: _rate(traps_ok[k], traps_n[k]) for k in sorted(traps_n)},
        "trap_preservation_all": _rate(sum(traps_ok.values()), sum(traps_n.values())),
    }
