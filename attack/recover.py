"""Step 6: recovery attack. The LLM guesses what each placeholder hid; guesses are checked
against the original record."""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from .data import PLACEHOLDER_RE
from .llm import LLMClient, inline_refs
from .tables import canonical_state

PROMPT_VERSION = "recover-v4"  # v4: neutral examples (v3 quoted values from evaluation records)
TYPES = ("name", "age", "state_or_region", "city", "year", "employer_or_facility", "occupation")
FUZZY_THRESHOLD = 85


class Guess(BaseModel):
    placeholder: str
    type: Literal["name", "age", "state_or_region", "city", "year", "employer_or_facility", "occupation"]
    guess: str
    reasoning: str
    confidence: Literal["low", "medium", "high"]


class Recovery(BaseModel):
    guesses: list[Guess]


SCHEMA = inline_refs(Recovery.model_json_schema())

SYSTEM_PROMPT = """You are a red-team analyst testing a de-identification system. This text was anonymized: identifying values were replaced by UPPERCASE placeholders in square brackets, such as [PATIENT_NAME], [US_STATE], [LOCATION], [DATE], [DOB] 1975, [ORGANIZATION], [OCCUPATION]. Lowercase speaker tags like [doctor] and [patient] are not placeholders (and may be swapped).

For each masked placeholder about the patient (their name, age, where they live, work or travelled, a city, a year, their employer, school or healthcare facility, their occupation), give your best guess of the original value and your reasoning. Use every clue: context, landmarks and local words, consistency between facts (a birth year plus a stated age gives the visit year), typical phrasing. Always give a concrete guess, even with low confidence; never answer with a placeholder or "unknown".

For each guess:
- placeholder: copy it exactly as it appears in the text, e.g. "[US_STATE]".
- type: name, age, state_or_region, city, year, employer_or_facility or occupation. A masked region, landmark, park or mountain range is state_or_region; a town is city.
- guess: the original value, e.g. "Ohio", "2019", "Tucson", "electrician".
- reasoning: one or two sentences citing the clues.
- confidence: low, medium or high.
One guess per distinct placeholder that actually appears in the text; skip placeholders about clinicians or other people, and weekdays, months, holidays and IDs. "Dax" and "Dragon" are the dictation software, not people. Never answer with a filler name such as John Doe: if nothing hints at the name, guess the most common name for the patient's sex and age. At most 15 guesses."""


def recover(client: LLMClient, text: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    return client.chat_json(messages, SCHEMA, "recovery", PROMPT_VERSION,
                            lambda d: Recovery.model_validate(d).model_dump())


# --- ground truth ---------------------------------------------------------------------------
# Words and placeholders only: punctuation tokens ("mrs." -> "mrs" ".") would anchor bad alignments.
_TOKEN = re.compile(PLACEHOLDER_RE.pattern + r"|\w+(?:['’]\w+)*")


def align_placeholders(anon_text: str, orig_text: str) -> dict[str, list[str]]:
    """Map each placeholder to the original text it replaced, by aligning the two versions."""
    a, o = _TOKEN.findall(anon_text), _TOKEN.findall(orig_text)
    matcher = SequenceMatcher(None, [t.lower() for t in a], [t.lower() for t in o], autojunk=False)
    truth: dict[str, list[str]] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        span = " ".join(o[j1:j2])
        for token in a[i1:i2]:
            if PLACEHOLDER_RE.fullmatch(token) and span:
                truth.setdefault(token, []).append(span)  # one entry per occurrence (repeats kept)
    return truth


# Fallback when a guessed placeholder cannot be aligned: compare with the Step 2 extraction
# of the original record.
_FALLBACK_FIELDS = {"age": ("age",), "state_or_region": ("state_or_region",), "city": ("city",),
                    "year": ("birth_year", "absolute_dates"), "occupation": ("occupation",),
                    "employer_or_facility": ("employer_or_facility",), "name": ()}
_TITLES = {"mr", "mrs", "ms", "miss", "dr", "doctor", "the", "a", "an", "of", "and"}
_GENERIC_ORG = {"hospital", "clinic", "medical", "center", "centre", "health", "healthcare", "university",
                "college", "school", "company", "inc", "llc", "corp", "group", "regional", "general"}


def _words(s: str, drop: set[str]) -> str:
    return " ".join(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in drop)


def is_match(kind: str, guess: str, truth: str) -> bool | None:
    """True/False, or None when this truth cannot be compared (e.g. a year guess vs 'next week')."""
    if kind == "age":
        g, t = re.search(r"\d{1,3}", guess), re.search(r"\d{1,3}", truth)
        return None if not t else bool(g) and abs(int(g.group()) - int(t.group())) <= 1
    if kind == "year":
        truth_years = set(re.findall(r"\b(?:19|20)\d{2}\b", truth))
        return None if not truth_years else bool(truth_years & set(re.findall(r"\b(?:19|20)\d{2}\b", guess)))
    if kind == "state_or_region":
        g_state, t_state = canonical_state(guess), canonical_state(truth)
        if g_state or t_state:
            return g_state == t_state
        g, t = _words(guess, _TITLES), _words(truth, _TITLES)  # regions: "North Cascades" vs "cascades"
        return None if not t else bool(g) and fuzz.token_set_ratio(g, t) >= FUZZY_THRESHOLD
    drop = _TITLES | (_GENERIC_ORG if kind == "employer_or_facility" else set())
    g, t = _words(guess, drop), _words(truth, drop)
    if not t:
        return None
    return bool(g) and fuzz.token_set_ratio(g, t) >= FUZZY_THRESHOLD


# Placeholder labels outside the attack's types (spec: name, age, place, year, employer, occupation).
OUT_OF_SCOPE_LABELS = {"MONTH", "WEEKDAY", "HOLIDAY", "ID", "ETHNICITY", "NATIONALITY", "CLINICIAN_NAME"}
FILLER_NAMES = {"john doe", "jane doe", "john smith", "jane smith", "patient", "the patient", "dax", "dragon",
                "unknown", "anonymous", "redacted", "mr x", "ms x", "name"}


def placeholder_label(key: str) -> str:
    m = re.match(r"\[([A-Z_]*?[A-Z])(?:_\d+)?(?::[^\]]*)?\]", key)
    return m.group(1) if m else ""


def score_guesses(recovery: dict, truth: dict[str, list[str]], orig_attrs: list[dict],
                  anon_placeholders: set[str] | None = None) -> list[dict]:
    """Ground truth: the aligned original span, else the original's extraction. Guesses for
    placeholders absent from the text or outside the attack's types are kept but unscorable;
    filler names ("John Doe") are scored as wrong, never as a lucky match on "john"."""
    scored, seen = [], set()
    guesses = sorted(recovery.get("guesses") or [], key=lambda g: _CONFIDENCE_RANK.get(g.get("confidence"), 3))
    for g in guesses:
        m = PLACEHOLDER_RE.search(g["placeholder"])
        key = m.group() if m else g["placeholder"].strip()
        if key in seen:  # one guess per placeholder (the most confident): repeats must not inflate the rate
            scored.append({**g, "truth": [], "truth_source": "duplicate_guess", "scorable": False, "correct": False})
            continue
        seen.add(key)
        if anon_placeholders is not None and key not in anon_placeholders:
            truths, source, verdicts = [], "placeholder_not_in_text", []
        elif placeholder_label(key) in OUT_OF_SCOPE_LABELS:
            truths, source, verdicts = truth.get(key, []), "out_of_scope", []
        else:
            truths, source = mode_truths(truth.get(key, []), g["type"]), "alignment"
            if not truths:
                truths = [a["value"] for a in orig_attrs if a["field"] in _FALLBACK_FIELDS[g["type"]]]
                source = "original_extraction"
            verdicts = [v for v in (is_match(g["type"], g["guess"], t) for t in truths) if v is not None]
            if verdicts and g["type"] == "name" and _words(g["guess"], _TITLES) in FILLER_NAMES:
                verdicts = [False]
        scored.append({**g, "truth": truths, "truth_source": source, "scorable": bool(verdicts),
                       "correct": any(verdicts)})
    return scored


_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _mode_key(span: str, kind: str) -> str:
    if kind == "year" and (m := re.search(r"\b(?:19|20)\d{2}\b", span)):
        return m.group()
    if kind == "age" and (m := re.search(r"\d{1,3}", span)):
        return m.group()
    return _words(span, set())


def mode_truths(spans: list[str], kind: str = "") -> list[str]:
    """A placeholder string can hide several values in one record ("[DATE]" for 2019, 2020 and 2021); a
    guess must match the value it hid most often (compared as a year for year guesses), not any of them."""
    if not spans:
        return []
    counts = Counter(_mode_key(s, kind) for s in spans)
    top = max(counts.values())
    keep = {k for k, c in counts.items() if c == top}
    out: list[str] = []
    for s in spans:
        if _mode_key(s, kind) in keep and s not in out:
            out.append(s)
    return out
_TYPE_RANK = {t: i for i, t in enumerate(("state_or_region", "city", "employer_or_facility", "occupation",
                                          "year", "age", "name"))}


def summarize_recovery(rows: list[dict]) -> dict:
    """rows: per-record dicts with 'encounter_id' and 'recovery' (the scored guesses)."""
    guesses = [{**g, "encounter_id": r["encounter_id"]} for r in rows for g in r.get("recovery") or []]

    def rates(gs: list[dict]) -> dict:
        scorable = [g for g in gs if g["scorable"]]
        correct = sum(g["correct"] for g in scorable)
        return {"n_guesses": len(gs), "n_scorable": len(scorable), "n_correct": correct,
                "recovery_rate": correct / len(scorable) if scorable else None}

    def rank(g: dict) -> tuple:
        return _CONFIDENCE_RANK[g["confidence"]], _TYPE_RANK[g["type"]]

    best = sorted((g for g in guesses if g["correct"]), key=rank)[:5]
    unverified = sorted((g for g in guesses if not g["scorable"] and g["confidence"] == "high"
                         and g["type"] != "name"), key=rank)[:5]
    # Names rarely have any clue in the text; the rate without them shows what context gives away.
    return {"overall": rates(guesses), "excluding_names": rates([g for g in guesses if g["type"] != "name"]),
            "by_type": {t: rates([g for g in guesses if g["type"] == t]) for t in TYPES},
            "best_recoveries": best, "unverified_high_confidence": unverified}
