"""Risk-adaptive transformations: placeholder, generalise, perturb, keep."""
from __future__ import annotations

import bisect
import hashlib
import os
import random
import re
import secrets

from .lexicon import number_to_words, words_to_number
from .rules import LABS, SPOKEN_YEAR
from .spans import RISK_CLASS, Span

ORG_PLACEHOLDER = {"healthcare": "[HEALTHCARE_FACILITY]", "school": "[SCHOOL]", "employer": "[EMPLOYER]"}
# geographic hierarchy: a country is generalised to its world region (travel history matters for
# infectious-disease differentials), a US state to "US state", a city to "city"
_REGIONS = {
    "North America": "canada mexico", "Caribbean": "bahamas barbados cuba dominican republic haiti jamaica trinidad puerto rico caribbean",
    "Central America": "belize costa rica el salvador guatemala honduras nicaragua panama central america",
    "South America": "argentina bolivia brazil chile colombia ecuador paraguay peru uruguay venezuela south america",
    "Europe": ("albania austria belarus belgium bosnia bulgaria croatia cyprus czech republic denmark england estonia "
               "finland france germany greece holland hungary iceland ireland italy latvia lithuania luxembourg malta "
               "moldova netherlands norway poland portugal romania russia scotland serbia slovakia slovenia spain "
               "sweden switzerland ukraine united kingdom wales europe"),
    "Middle East": "iran iraq israel jordan kuwait lebanon qatar saudi arabia syria turkey united arab emirates yemen middle east",
    "Africa": ("algeria cameroon congo egypt ethiopia ghana kenya liberia libya madagascar mali morocco mozambique niger "
               "nigeria rwanda senegal somalia south africa sudan tanzania tunisia uganda zambia zimbabwe africa"),
    "Asia": ("afghanistan armenia bangladesh cambodia china hong kong india indonesia japan kazakhstan korea south korea "
             "north korea laos malaysia mongolia myanmar nepal pakistan philippines singapore sri lanka taiwan thailand "
             "uzbekistan vietnam asia"),
    "Oceania": "australia fiji new zealand",
}
REGION_OF = {c: region for region, names in _REGIONS.items()
             for c in re.findall(r"czech republic|dominican republic|costa rica|el salvador|puerto rico|saudi arabia|"
                                 r"united arab emirates|united kingdom|hong kong|south korea|north korea|sri lanka|"
                                 r"south africa|new zealand|south america|central america|middle east|[a-z]+", names)}


# most certain first: a 4-digit year anywhere (ISO dates start with it), a spoken year, then a
# two-digit year at the end of a numeric date ("7/14/13", "'85": century unknown)
_YEAR_FORMS = [re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)"), re.compile(SPOKEN_YEAR, re.I),
               re.compile(r"'\d{2}\b|(?<=[/.-])\d{2}$")]


def year_in(surface: str) -> tuple[str, int | None] | None:
    """The calendar year written inside a date (as written) and its numeric value when certain."""
    surface = surface.strip()
    if re.fullmatch(r"\d{8}", surface):  # compact YYYYMMDD / MMDDYYYY
        y = surface[:4] if surface[:2] in ("19", "20") else surface[4:]
        return (y, int(y)) if y[:2] in ("19", "20") else None
    m = next((hits[-1] for rx in _YEAR_FORMS if (hits := list(rx.finditer(surface)))), None)
    if not m:
        return None
    s = m.group(0)
    if re.fullmatch(r"\d{4}", s):
        return s, int(s)
    words = s.lower().split()
    if words[0] in ("nineteen", "twenty") and len(words) > 1:
        rest = words_to_number(" ".join(words[1:]).replace("oh ", ""))
        return s, (1900 if words[0] == "nineteen" else 2000) + int(rest or 0)
    if words[0] == "two":
        return s, 2000 + int(words_to_number(" ".join(words[2:]).replace("and ", "")) or 0) if len(words) > 2 else 2000
    return s, None  # two-digit year: century unknown


def birth_year(surface: str, reference_year: int = 2026) -> int | None:
    """Calendar year of a birth date, including two-digit years ("3/4/78" -> 1978, "12/01/09" -> 2009)."""
    yr = year_in(surface)
    if not yr:
        return None
    if yr[1] is not None:
        return yr[1]
    m = re.search(r"\d{2}", yr[0])
    if not m:
        return None
    y = int(m.group(0))
    return 2000 + y if 2000 + y <= reference_year else 1900 + y


def age_band(age: float, width: int = 10, top: int = 90, mode: str = "band") -> str:
    a = int(age)
    if mode == "coarse":  # last-resort generalisation for records that are otherwise unique
        return "<18" if a < 18 else "18-64" if a < 65 else "65+"
    if a >= top:
        return f"{top}+"
    if a < 1:
        return "<1"
    if a < 20:  # finer bands for children (paediatric codes depend on it)
        for lo, hi in ((1, 4), (5, 9), (10, 14), (15, 19)):
            if lo <= a <= hi:
                return f"{lo}-{hi}"
    lo = a // width * width
    return f"{lo}-{lo + width - 1}"


def adult_band(age: float, width: int = 10, top: int = 90, exact_below: int = 18) -> str | None:
    """Age policy 'band': None = publish exactly (paediatric ages), else "18-19", "20-29" ... "80-89", "90+"."""
    a = int(age)
    if a < exact_below:
        return None
    if a >= top:
        return f"{top}+"
    if a < 20:
        return f"{exact_below}-19"
    lo = 20 + (a - 20) // width * width
    return f"{lo}-{min(lo + width - 1, top - 1)}"


_BMI_CLASSES = [(18.5, "<18.5"), (25, "18.5-24.9"), (30, "25-29.9"), (35, "30-34.9"), (40, "35-39.9")]


def anthro_interval(meta: dict, pol: dict) -> str:
    """Height/weight -> fixed-width interval in the unit as written; BMI -> WHO class (thresholds kept)."""
    kind, v = meta["kind"], float(meta["value"])
    if kind == "bmi":
        return "[" + next((lab for lim, lab in _BMI_CLASSES if v < lim), ">=40") + "]"
    if meta.get("unit") == "ftin":
        w = pol.get("in_width", 2)
        lo = int(v // w * w)
        return f"[{lo // 12} ft {lo % 12} in - {(lo + w) // 12} ft {(lo + w) % 12} in]"
    if meta.get("unit") == "m":
        lo = int(v * 100 // pol.get("cm_width", 5) * pol.get("cm_width", 5))
        return f"[{lo / 100:.2f}-{(lo + pol.get('cm_width', 5)) / 100:.2f}]"
    w = pol.get({"kg": "kg_width", "lb": "lb_width"}.get(meta.get("unit"), "cm_width"), 5)
    lo = int(v // w * w)
    return f"[{lo}-{lo + w}]"


def perturb_lab(lab: str, value: float, decimals: int, seed: str, max_rel: float) -> float | None:
    """Shift a lab value slightly *without leaving its clinical category*.

    Categories are delimited by guideline decision thresholds (e.g. HbA1c 5.7 / 6.5 / 7 / 8 / 9 %),
    so the perturbed value supports exactly the same diagnosis/ICD-10 code. Returns None if no
    distinct value exists inside the category at the reported precision.
    """
    thresholds, (lo, hi) = LABS[lab][1], LABS[lab][2]
    step = 10 ** -decimals
    cat = bisect.bisect_right(thresholds, value)
    # never exceed the configured relative change: at coarse precision small values cannot move
    # (TSH 0.1 at one decimal would need +100 %), and are then kept rather than distorted
    max_steps = int(abs(value) * max_rel / step + 1e-9)
    cands = []
    for d in range(1, max_steps + 1):
        for sign in (1, -1):
            nv = round(value + sign * d * step, decimals)
            if lo <= nv <= hi and bisect.bisect_right(thresholds, nv) == cat and nv != value \
                    and abs(nv - value) <= max_rel * abs(value) + 1e-9:
                cands.append(nv)
    if not cands:
        return None
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    return rng.choice(sorted(set(cands)))


class Transformer:
    def __init__(self, cfg: dict):
        from .lexicon import load_lexicon

        self.cfg = cfg
        self.pol = cfg["policies"]
        self.places = load_lexicon().places
        # The perturbation key must not be derivable from published data (config seed + encounter
        # id would let an attacker recompute and invert it). MEDANON_SECRET makes runs reproducible.
        self.secret = cfg.get("perturbation_secret") or os.environ.get("MEDANON_SECRET") or secrets.token_hex(16)
        self.secret_source = "config" if cfg.get("perturbation_secret") else \
            "env MEDANON_SECRET" if os.environ.get("MEDANON_SECRET") else "random per run (not stored)"

    def apply(self, span: Span, text: str, encounter_id: str, person_placeholder: str | None,
              overrides: dict | None = None) -> tuple[str | None, str, str]:
        """Return (replacement or None to keep, action, explanation)."""
        label = span.label
        pol = {**self.pol.get(label, {"action": "placeholder"}), **(overrides or {}).get(label, {})}
        action = pol.get("action", "placeholder")
        risk = RISK_CLASS.get(label, "identifier")
        if action == "keep":
            return None, "keep", f"{risk}; policy keeps {label}"
        if action == "redact":
            return "", "redact", f"{risk}; removed entirely by policy"
        if label == "PERSON":
            ph = person_placeholder or "[PERSON]"
            return ph, "pseudonymize", f"{risk}: personal name replaced by consistent role placeholder {ph}"
        if label == "AGE" and action == "exact":
            age, top = span.meta.get("age", 0), pol.get("top_code", 90)
            if age >= top:
                return f"{top}+", "generalize", f"{risk}: age of {top} or more aggregated as {top}+ (HIPAA Safe Harbor)"
            return None, "keep", (f"{risk}: exact age up to {top - 1} kept by policy (clinically relevant; raises "
                                  f"re-identification risk together with sex and ICD-10 codes)")
        if label == "ANTHROPOMETRIC" and action == "generalize":
            if span.meta.get("exact_reason"):
                return None, "keep", f"{risk}: kept exactly - {span.meta['exact_reason']}"
            return anthro_interval(span.meta, pol), "generalize", \
                f"{risk}: {span.meta['kind']} generalised to an interval (BMI: WHO class), exact value not published"
        if label == "FAMILY_SIZE" and action == "generalize":
            top = pol.get("exact_up_to", 2)
            if span.meta.get("count", 0) <= top:
                return None, "keep", f"{risk}: small family size ({top} or fewer) is common, kept"
            return f"[{top + 1}+]", "generalize", f"{risk}: number of children above {top} generalised to {top + 1}+"
        if label == "AGE" and action == "band":
            band = adult_band(span.meta.get("age", 0), pol.get("band_width", 10), pol.get("top_code", 90),
                              pol.get("exact_below", 18))
            if band is None:
                return None, "keep", f"{risk}: paediatric age kept exactly (dosing, development, paediatric codes)"
            if band.endswith("+"):
                return band, "generalize", f"{risk}: age of {band[:-1]} or more aggregated as {band} (HIPAA Safe Harbor)"
            return pol.get("format", "[{band}]").format(band=band), "generalize", f"{risk}: adult age generalised to {band}"
        if label == "DOB" and pol.get("keep_year") and self.pol.get("AGE", {}).get("action") == "band":
            # age-band policy: a published adult birth year gives the exact age back (visit year - birth year);
            # decide from the year itself (4-digit, 2-digit, spoken), not from whether an age was also detected
            by = birth_year(text[span.start:span.end], pol.get("reference_year", 2026))
            if by is None or pol.get("reference_year", 2026) - by >= self.pol["AGE"].get("exact_below", 18):
                pol = {**pol, "keep_year": False}
        if label in ("DOB", "DATE") and pol.get("keep_year") and action in ("generalize", "placeholder"):
            surface = text[span.start:span.end]
            ph = pol.get("placeholder", f"[{label}]")
            yr = year_in(surface)
            too_old = yr and yr[1] is not None and label == "DOB" and pol.get("reference_year") \
                and pol["reference_year"] - yr[1] >= 90
            if yr and not too_old:
                if yr[0].strip() == surface.strip():
                    return None, "keep", f"{risk}: only a calendar year, kept by policy"
                return f"{ph} {yr[0]}", "generalize", f"{risk}: day and month removed, calendar year kept by policy"
            if too_old:
                return ph, "pseudonymize", f"{risk}: birth year removed because it reveals an age of 90 or more"
        if label == "AGE" and action == "generalize":
            band = age_band(span.meta.get("age", 0), pol.get("band_width", 10), pol.get("top_code", 90), pol.get("mode", "band"))
            return pol.get("format", "[{band}]").format(band=band), "generalize", \
                f"{risk}: exact age generalised to the {band} band"
        if label == "OCCUPATION" and action == "generalize":
            cat = span.meta.get("category", "other")
            return ("[OCCUPATION]" if cat == "other" else f"[OCCUPATION: {cat}]"), "generalize", \
                f"{risk}: specific occupation generalised to its category '{cat}' (keeps exposure/physical-demand signal)"
        if label == "LAB_VALUE":
            if action == "generalize":  # value -> its guideline category interval (defeats tolerant linkage)
                lab, val = span.meta["lab"], span.meta["value"]
                th, (lo, hi) = LABS[lab][1], LABS[lab][2]
                i = bisect.bisect_right(th, val)
                band = f"<{th[0]:g}" if i == 0 else f">={th[-1]:g}" if i == len(th) else f"{th[i - 1]:g}-{th[i]:g}"
                return f"[{band}]", "generalize", (f"{risk}: {lab} generalised to its clinical category {band} "
                                                   f"(thresholds {th}); protects against approximate-value linkage")
            if action != "perturb":
                return None, "keep", "lab value kept by policy"
            lab, val, dec = span.meta["lab"], span.meta["value"], span.meta["decimals"]
            nv = perturb_lab(lab, val, dec, f"{self.secret}|{encounter_id}|{lab}|{val}", pol.get("max_relative", 0.03))
            if nv is None:
                return None, "keep", (f"{lab} value kept: no value within ±{pol.get('max_relative', 0.03):.0%} at the "
                                      f"reported precision stays in the same clinical category")
            rep = f"{nv:.{dec}f}" if span.meta.get("form") == "digits" else number_to_words(nv, dec)
            return rep, "perturb", (f"{risk}: {lab} perturbed by <={int(pol.get('max_relative', 0.03) * 100)}% "
                                    f"within the same clinical category (thresholds {LABS[lab][1]})")
        if label == "ORGANIZATION":
            ph = ORG_PLACEHOLDER.get(span.meta.get("org_type", ""), pol.get("placeholder", "[ORGANIZATION]"))
            return ph, "pseudonymize", f"{risk}: named organisation replaced by {ph}"
        if label == "LOCATION" and pol.get("hierarchy", True):
            key = re.sub(r"^(?:the)\s+", "", text[span.start:span.end].lower()).strip()
            ptype = span.meta.get("place_type") or self.places.get(key)
            if key in REGION_OF:
                ph = f"[LOCATION: {REGION_OF[key]}]"
                return ph, "generalize", f"{risk}: country generalised to world region {ph} (keeps travel-history signal)"
            if ptype in ("state", "city"):
                ph = "[US_STATE]" if ptype == "state" else "[CITY]"
                return ph, "pseudonymize", f"{risk}: {ptype} replaced by {ph}"
        ph = pol.get("placeholder", f"[{label}]")
        return ph, "pseudonymize", f"{risk}: replaced by {ph}"
