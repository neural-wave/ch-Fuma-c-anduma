"""Step 5: turn verified attributes into bits, lambda and re-identification risk."""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict

from . import tables
from rapidfuzz import fuzz

from .verify import HALLUCINATION_REASONS, normalize

N_POP = 335e6  # US population
LAMBDA_FLOOR = 1e-12
MAX_OTHER_FACTS = 3
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def risk_metrics(ps: list[float]) -> dict:
    """bits = Σ -log2 p;  λ = (N-1)·Π p  (expected look-alikes);  risk = (1 - e^-λ)/λ, the chance
    of picking the right person among the 1 + Poisson(λ) candidates who share every attribute."""
    bits = sum(-math.log2(p) for p in ps)
    lam = max((N_POP - 1) * 2.0 ** -bits, LAMBDA_FLOOR)
    return {"bits": bits, "lambda": lam, "risk": -math.expm1(-lam) / lam}


def attribute_p(a: dict, event_year: int | None) -> tuple[str, float, str, str]:
    """-> (category, p, p_source, note) for one verified attribute."""
    field, value = a["field"], a["value"]
    if field == "age":
        rng = tables.parse_age(value)
        if not rng:
            return "age", 1.0, "constant", "unparsed age"
        lo, hi = rng
        return "age", tables.age_range_share(lo, hi), "census", f"age {lo}" if lo == hi else f"age {lo}-{hi}"
    if field == "birth_year":
        m = _YEAR.search(value)
        if not m:
            return "age", 1.0, "constant", "unparsed birth year"
        born = int(m.group())
        if event_year and event_year > born:  # birth year + event year -> exact age, counted once
            return "age", tables.age_range_share(*(2 * [event_year - born])), "census", \
                f"born {born} + event year {event_year} -> age {event_year - born}"
        age = tables.CENSUS_YEAR - born
        return "age", tables.age_range_share(age, age), "census", f"born {born}: one-year birth cohort"
    if field == "sex":
        return "sex", 0.5, "constant", value.lower()
    if field == "state_or_region":
        p, source, note = tables.place_p(value)
        return "state_or_region", p, source, note
    if field == "city":
        return "city", tables.CITY_P, "constant", value.lower()
    if field == "occupation":
        p, source, note = tables.occupation_p(value)
        return "occupation", p, source, note
    if field == "employer_or_facility":
        p, note = tables.employer_p(value)
        return "employer_or_facility", p, "constant", note
    if field == "ethnicity":
        p, source, note = tables.ethnicity_p(value)
        return "ethnicity", p, source, note
    if field == "nationality":
        p, source, note = tables.nationality_p(value)
        return "nationality", p, source, note
    if field == "absolute_dates":
        p, precision = tables.date_p(value)
        return "absolute_date", p, "constant", f"{precision} precision"
    if field == "family_structure":
        p, note = tables.family_p(value)
        return "family_structure", p, "bucket", note
    if field == "other_facts" and FAMILY_HINT.search(value):
        # household and family facts are priced (and de-duplicated) as family structure
        p, note = tables.family_p(value)
        return "family_structure", p, "bucket", f"{note} (from other_facts)"
    if field == "other_facts" and not a.get("inferred_place") and (state := tables.canonical_state(value)):
        # "hikes in Washington": the fact ties the patient to a state, priced like one
        return "state_or_region", tables.STATE_POP[state] / sum(tables.STATE_POP.values()), "census", \
            f"{state} (from other_facts)"
    if field == "other_facts" and (hit := tables.lifestyle_p(value)):
        return "other_facts", hit[0], "table", hit[1]
    if field == "other_facts":
        rarity = a.get("rarity") or "common"
        return "other_facts", tables.RARITY_P.get(rarity, 0.1), "bucket", rarity
    raise ValueError(f"unknown field {field}")


def score_record(verified: list[dict]) -> dict:
    """Price every attribute, keep one per category (the rarest; the 3 rarest other_facts), score."""
    years = [int(y) for a in verified if a["field"] == "absolute_dates" for y in _YEAR.findall(a["value"])]
    event_year = max(years, default=None)
    attrs = []
    for a in verified:
        category, p, source, note = attribute_p(a, event_year)
        p = min(max(p, 1e-12), 1.0)
        attrs.append({**a, "category": category, "p": p, "p_source": source, "note": note,
                      "bits": -math.log2(p), "counted": False})
    by_category = defaultdict(list)
    for a in attrs:
        by_category[a["category"]].append(a)
    decided: list[dict] = []  # counted attributes of the other categories
    for category in sorted(by_category, key=lambda c: c == "other_facts"):  # other_facts last
        group = sorted(by_category[category], key=lambda a: a["p"])
        counted: list[dict] = []
        for a in group:
            if len(counted) == (MAX_OTHER_FACTS if category == "other_facts" else 1):
                break
            dup = None
            if category == "other_facts":
                dup = next((c for c in counted if same_fact(a, c) or restates(a, c, min_shared=2)
                            or same_theme(a, c)), None) or next((c for c in decided if restates(a, c)), None)
            if dup:
                a["duplicate_of"] = dup["value"]
                continue
            a["counted"] = True
            counted.append(a)
        if category != "other_facts":
            decided += counted
    return {**risk_metrics([a["p"] for a in attrs if a["counted"]]), "attributes": attrs}


_STOP = {"with", "from", "that", "this", "have", "has", "been", "very", "their", "about", "lives", "live"}


def same_fact(a: dict, b: dict) -> bool:
    """Two other_facts restating one fact: overlapping evidence and a shared word stem
    ("hiking" / "hikes regularly"). Two facts quoted from one sentence keep distinct stems."""
    ea, eb = normalize(a.get("evidence") or ""), normalize(b.get("evidence") or "")
    if not ea or not eb or not (ea in eb or eb in ea or fuzz.partial_ratio(ea, eb) >= 90):
        return False

    def stems(v: str) -> set[str]:
        return {w[:3] for w in re.findall(r"[a-z]{4,}", v.lower()) if w not in _STOP}
    return bool(stems(a["value"]) & stems(b["value"]))


FAMILY_HINT = re.compile(r"\b(?:siblings?|brothers?|sisters?|children|child|kids?|sons?|daughters?|wife|husband|"
                         r"spouse|partner|married|widow\w*|divorc\w*|lives? alone|lives? with|roommates?|"
                         r"grand(?:children|kids|sons?|daughters?)|twins?|triplets?|expecting|of us)\b", re.I)
_GENERIC = {"work", "works", "worker", "working", "with", "from", "that", "this", "have", "been", "very", "their",
            "about", "since", "years", "year", "times", "week", "weekly", "daily", "every", "often", "recently",
            "regularly", "frequently", "patient", "also", "uses", "takes", "plays", "does", "goes", "makes", "gets",
            "home", "house", "lives", "live", "living", "family", "job"}


_THEMES = {"current smoker": "smoking", "former smoker": "smoking"}


def same_theme(a: dict, b: dict) -> bool:
    """Two facts priced from the same lifestyle-table entry ("smokes 1.5 packs" / "started smoking in high school")."""
    if a["p_source"] != "table" or b["p_source"] != "table":
        return False
    return _THEMES.get(a["note"], a["note"]) == _THEMES.get(b["note"], b["note"])


def restates(a: dict, other: dict, min_shared: int = 1) -> bool:
    """`a` restates `other` when they share informative word stems: one for an attribute of another
    category ("works as a lecturer" / occupation "lecturer"), two between other_facts quoted from
    different places ("takes nature photographs" / "enjoys nature photography")."""
    def stems(v: str) -> set[str]:
        return {w[:4] for w in re.findall(r"[a-z]{4,}", v.lower()) if w not in _GENERIC}
    return len(stems(a["value"]) & stems(other["value"])) >= min_shared


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("failed")]
    bits = [r["bits"] for r in ok]
    leak: dict[str, float] = defaultdict(float)
    for r in ok:
        for a in r["attributes"]:
            if a["counted"]:
                leak[a["category"]] += a["bits"]
    n_extracted = sum(r["n_extracted"] for r in ok)
    reasons = Counter(d["reason"] for r in ok for d in r["dropped"])
    n_hallucinated = sum(n for reason, n in reasons.items() if reason in HALLUCINATION_REASONS)
    return {
        "n_records": len(rows),
        "n_failed": len(rows) - len(ok),
        "mean_risk": statistics.fmean(r["risk"] for r in ok) if ok else None,
        "mean_bits": statistics.fmean(bits) if ok else None,
        "median_bits": percentile(bits, 0.5),
        "p90_bits": percentile(bits, 0.9),
        "pct_likely_unique": 100.0 * sum(r["lambda"] < 1 for r in ok) / len(ok) if ok else None,
        "top_leaking_attributes": dict(sorted(leak.items(), key=lambda kv: -kv[1])),
        "hallucination_rate": n_hallucinated / n_extracted if n_extracted else 0.0,
        "dropped_by_reason": dict(reasons),
        "mean_masking_density": statistics.fmean(r["masking_density"] for r in rows) if rows else None,
    }


def compare(anon: dict, orig: dict) -> dict:
    gain = (1 - anon["mean_risk"] / orig["mean_risk"]) if orig.get("mean_risk") else None
    both_bits = None not in (orig.get("mean_bits"), anon.get("mean_bits"))
    removed = orig["mean_bits"] - anon["mean_bits"] if both_bits else None
    return {"privacy_gain": gain, "bits_removed": removed}
