"""Step 3: keep only attributes whose evidence is really in the text."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .data import PLACEHOLDER_RE
from .extract import SCALAR_FIELDS
from .tables import canonical_state, date_p

QUOTE_THRESHOLD = 90
VALUE_THRESHOLD = 80
# Fields whose value must appear in its own evidence: catches values the model inferred
# (e.g. a state guessed from a landmark), which belong to the recovery attack instead.
LITERAL_FIELDS = {"birth_year", "state_or_region", "city", "employer_or_facility", "absolute_dates"}
HALLUCINATION_REASONS = {"missing_evidence", "evidence_not_found", "value_not_in_evidence"}
# other_facts the prompt forbids but models still return. Diagnoses are a floor the pipeline must
# keep (reported via n_icd_codes), and names are out of the extraction schema.
MEDICAL_RE = re.compile(
    r"diagnos|disease|syndrome|disorder|cancer|tumou?r|carcinoma|\w+itis\b|\w+osis\b|\b(?!diploma)\w+oma\b|"
    r"diabet|hypertens|hypotens|transplant|dialysis|depress|anxiety|asthma|\bcopd|arthritis|thyroid|"
    r"cardiac|\bheart\b|kidney|renal\b|\bliver\b|\blungs?\b|surgery|surgical|\boperation|medication|"
    r"prescri|\bmg\b|\bpain(?:s|ful)?\b|\bache|fracture|injur|sprain|pregnan|vaccin|therapy|treatment|symptom|"
    r"infection|allerg|failure|stroke|seizure|migraine|obes|cholesterol|blood pressure|\blabs?\b|\bmri\b|"
    r"ct scan|biopsy|chemo|insulin|hospitali[sz]|admitted|emergency room|history of|murmur|fibrillation|"
    r"\ba-?fib\b|arrhythm|palpitation|mammogra|colonoscop|screening|ejection fraction|\b(?:ekg|ecg)\b|"
    r"echocardio|ultrasound|\bscans?\b|imaging|\bdoses?\b|\bpills?\b|tablet|inhaler|nebuli|\bcpap\b|apnea|"
    r"\brash\b|swelling|fever|cough|nausea|vomit|dizz|fatigue|\bbmi\b|edema|wheez|numbness|tingl|bleed|"
    r"wound|ulcer|lesion|\bcysts?\b|nodule|polyp|hernia|\bgout\b|anemi|dementia|alzheimer|parkinson|lupus|"
    r"\bhiv\b|glucose|\ba1c\b|statin|antibiotic|steroid|follow-?up|check-?up|referr|physical exam|"
    r"blood (?:sugar|work|test)|weight (?:gain|loss)|\w+ectomy|\w+otomy|\w+ostomy|\w+plasty|\w+scopy|\bvalves?\b|"
    r"replacement|reconstruction|\bstents?\b|bypass|(?<!auto )repair\b|\bpounds\b|\blbs\b|\bweight\b|supplement|"
    r"vitamin|coq10|\bthc\b|\bcbd\b|gummies|\bcreams?\b|ointment|\bexams?\b|self-exam|menstruat|menopaus|"
    r"menarche|breathing|\bankles?\b|\blips?\b|\w+(?:mycin|cillin|floxacin|cycline|pril|olol|sartan|statin|"
    r"prazole|tidine|zepam|oxetine|triptan|formin)\b|\bprior to (?:dental|procedures?)|epilep|gallstone|\w+lith\w*|"
    r"\bfusion\b|nause|\brash(?:es)?\b|\bibs\b|\bpcos\b|tonsil|(?:ekg|ecg)s\b|counsel(?:or|ing)|therapist|psychiatr|"
    r"psycholog|wheelchair|\bwalker\b|\bcanes?\b|crutch|\bscars?\b|\bcramps?\b|\bgel\b|medicated|\bmeds?\b|pill ?box|"
    r"sensitive skin|eczema|psoria|\bacne\b|dermat|remed(?:y|ies)", re.I)
# A fact whose informative part is a masked value ("traveled to a US state" quoted from "[US_STATE]").
MASKED_REFERENT_RE = re.compile(r"\b(?:an?|some|unspecified|unknown|undisclosed|unnamed|certain) (?:us |u\.s\. )?"
                                r"(?:state|city|town|location|place|country|organization|company|employer|facility|"
                                r"hospital|clinic|store|restaurant)\b|\b(?:redacted|masked|unspecified|undisclosed)\b",
                                re.I)
# Deterministic fallback for the patient's age, used only when the LLM extracted none: the patient
# introduction ("is a 46 -year-old female", "a [40-49]-year-old male"). A sex/person word must follow,
# so relatives ("my 45-year-old husband") never match.
AGE_INTRO_RE = re.compile(r"(\[\d{1,3}-\d{1,3}\]|\[?\d{1,3}\+\]?|\d{1,3}(?:\s\d{1,3})?)\s*-?\s*(?:years?|yrs?)"
                          r"\s*-?\s*old\s+(?:(?:white|black|hispanic|latino|latina|asian|caucasian|african american|"
                          r"elderly|pleasant)\s+)?"
                          r"(?:male|female|man|woman|boy|girl|gentleman|lady|patient)\b", re.I)


def fallback_age(text: str) -> dict | None:
    m = AGE_INTRO_RE.search(text)
    if not m:
        return None
    value = re.sub(r"[\[\]]", "", m.group(1)).replace(" ", "-")
    return {"field": "age", "value": value, "evidence": m.group(0), "fallback": True,
            "rationale": "deterministic fallback: patient age in the introduction pattern"}


NAME_HINT_RE = re.compile(r"\b(?:(?:full|first|last|sur|maiden|nick) ?name|named|name is|called)\b", re.I)
_ORG_WORDS = (r"Hospital|Clinic|Center|Centre|Orthopedics|Medical|Health|Inc|Shop|School|College|University|"
              r"Church|Company|Farm|Bank|Club|Team|Army|Navy|Marines?|Corps")
NAME_RE = re.compile(rf"(?!.*\b(?:{_ORG_WORDS})\b)"
                     r"(?!(?:Mount|Mt|Lake|River|Park|Camp|Fort|Saint|St|New|North|South|East|West|Upper)\b)"
                     r"(?:(?:mr|mrs|ms|miss|dr)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}")

_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_EDGE = " \t\n.,;:!?\"'()[]-"


def normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s.translate(_TYPOGRAPHY).lower())
    s = re.sub(r" ([,.;:!?])", r"\1", s)  # the dialogues are tokenized: "tech ." -> "tech."
    return s.strip(_EDGE)


def quote_in_text(quote: str, norm_text: str, fuzzy: bool = True) -> bool:
    """Models often elide ("a sweet tooth ... i love jelly beans"): every fragment must then be found."""
    fragments = [normalize(f) for f in re.split(r"\s*(?:\.\.\.|…)\s*", quote)]
    fragments = [f for f in fragments if len(f) >= 3]
    return bool(fragments) and all(
        f in norm_text or (fuzzy and fuzz.partial_ratio(f, norm_text) >= QUOTE_THRESHOLD) for f in fragments)


def clean_value(value: str) -> str:
    """Remove placeholders from a value, keeping visible residue ("[DATE] 2021" -> "2021")."""
    v = re.sub(r"\[[A-Z][A-Z_]*(?:_\d+)?:\s*([^\]]*)\]", r"\1", value)
    return re.sub(r"\s+", " ", PLACEHOLDER_RE.sub(" ", v)).strip(" ,;:-")


def flatten(extraction: dict) -> list[dict]:
    items = [{"field": f, **(extraction.get(f) or {})} for f in SCALAR_FIELDS]
    items += [{"field": "absolute_dates", **a} for a in extraction.get("absolute_dates") or []]
    items += [{"field": "other_facts", **a} for a in extraction.get("other_facts") or []]
    return [a for a in items if (a.get("value") or "").strip()]


def verify(extraction: dict, text: str, icd_descriptions: tuple[str, ...] = (),
           fuzzy: bool = True, icd_external: tuple[str, ...] = ()) -> tuple[list[dict], list[dict], int]:
    """-> (kept attributes, dropped attributes with a reason, number extracted). fuzzy=False requires
    the evidence verbatim: a quote with a value the anonymizer masked can then never match."""
    norm_text = normalize(text)
    kept, dropped = [], []
    items = flatten(extraction)
    for a in items:
        value, evidence = clean_value(a["value"]), (a.get("evidence") or "").strip()
        if not re.search(r"[A-Za-z0-9]", value):
            reason = "placeholder_value"
        elif not evidence:
            reason = "missing_evidence"
        elif not quote_in_text(evidence, norm_text, fuzzy):
            reason = "evidence_not_found"
        elif a["field"] in LITERAL_FIELDS and not _value_in(value, evidence):
            reason = "value_not_in_evidence"
        elif a["field"] in ("other_facts", "family_structure") and (is_medical(value, icd_descriptions) or (
                a["field"] == "family_structure" and is_medical(evidence))):  # family history of disease
            reason = "medical_fact"
        elif a["field"] == "other_facts" and PLACEHOLDER_RE.search(evidence) and MASKED_REFERENT_RE.search(value):
            reason = "masked_referent"
        elif a["field"] == "other_facts" and restates_icd(value, icd_external):
            reason = "icd_floor"  # e.g. "plays golf" next to the code "Activity, golf": kept by the task rules
        elif a["field"] == "other_facts" and (NAME_HINT_RE.search(value) or (
                NAME_RE.fullmatch(value) and not canonical_state(value))):
            reason = "name_not_scored"
        elif a["field"] == "absolute_dates" and date_p(value)[1] == "unparsed":
            reason = "not_absolute_date"  # "last weekend", "4 days ago": relative times, no information
        else:
            state = canonical_state(value) if a["field"] == "other_facts" else None
            if state and state not in norm_text:  # the extractor inferred a masked place: not a stated fact
                kept.append({**a, "value": value, "inferred_place": state})
            else:
                kept.append({**a, "value": value})
            continue
        dropped.append({**a, "reason": reason})
    return kept, dropped, len(items)


_ICD_GENERIC = {"activity", "activities", "other", "unspecified", "person", "injured", "injury", "initial",
                "encounter", "subsequent", "with", "without", "involving", "due", "place", "occurrence", "status",
                "unknown", "type", "mode", "transport", "nontraffic"}


# Words a fact may use for the concept an external-cause code names ("rear-ended" for a vehicle accident).
_ICD_CONCEPTS = {
    r"vehicle|motor|traffic|car\b|truck|motorcycle|pedestrian": r"accident|crash|collision|rear|wreck|vehicle|car\b|"
                                                                 r"truck|driv|traffic|motorcycle|\bhit\b|struck|"
                                                                 r"four-way|intersection|stop sign|stoplight",
    r"\bfall|slipping|tripping|stumbling": r"\bfell\b|\bfall|slip|trip|stumbl",
}


def restates_icd(value: str, icd_external: tuple[str, ...]) -> bool:
    """A fact an external-cause/activity code already states: a shared word ("golf" / "golfing") or
    a synonym of its concept ("rear-ended" for a motor-vehicle accident)."""
    stems = {w[:4] for w in re.findall(r"[a-z]{4,}", value.lower())}
    for d in icd_external:
        if stems & {w[:4] for w in set(re.findall(r"[a-z]{4,}", d.lower())) - _ICD_GENERIC}:
            return True
        if any(re.search(code_re, d, re.I) and re.search(fact_re, value, re.I)
               for code_re, fact_re in _ICD_CONCEPTS.items()):
            return True
    return False


def is_medical(value: str, icd_descriptions: tuple[str, ...] = ()) -> bool:
    return bool(MEDICAL_RE.search(value)) or any(
        fuzz.token_set_ratio(value.lower(), d.lower()) >= 85 for d in icd_descriptions)


def _value_in(value: str, evidence: str) -> bool:
    v, e = normalize(value), normalize(evidence)
    return bool(v) and (v in e or fuzz.partial_ratio(v, e) >= VALUE_THRESHOLD)
