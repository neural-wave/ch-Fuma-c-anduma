"""Clinical-context guard: decides whether a candidate identifier is actually medical content.

Rationale: generic NER confuses eponyms (Parkinson's, Bell's palsy, Baker's cyst), drug brand
names (Lasix, Allegra), clinical abbreviations (COPD, CBC) and some place-named diseases (Lyme)
with persons, organisations or locations. Removing them destroys exactly the information the
ICD-10 utility metrics measure, so every PERSON/ORG/LOCATION/NATIONALITY candidate is checked here.
The per-record ICD-10 descriptions (available at inference time) act as extra evidence; the guard
also works without them (bonus task "no ICD codes").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import MEDICAL_HEADS, SYSTEM_TERMS, Lexicon
from .spans import Span

GUARDED_LABELS = {"PERSON", "ORGANIZATION", "LOCATION", "NATIONALITY"}
INSTRUMENT_HEADS = {"scale", "score", "criteria", "index", "questionnaire", "classification", "inventory", "test",
                    "tests", "maneuver", "manoeuvre", "sign"}
KNOWN_ORG_ACRONYMS = {"usa", "us", "uk", "nyu", "ucla", "usc", "mit", "fbi", "cia", "ups", "fedex", "ibm",
                      "nasa", "nfl", "nba", "mlb", "nhl", "va", "ymca", "cvs"}
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9'\-]*")  # "SpO2", "B12" stay one word
# formulation / device suffixes that accompany drug brands ("ProAir HFA", "Toprol XL", "Bactrim DS")
DOSAGE_FORMS = {"hfa", "xr", "er", "sr", "cr", "xl", "ds", "odt", "la", "cd", "dr", "hct", "plus", "respiclick",
                "diskus", "ellipta", "flexpen", "kwikpen", "solostar", "pen", "inhaler", "tablet", "tablets",
                "capsule", "capsules", "patch", "cream", "ointment", "gel", "spray", "drops", "injection", "mg", "mcg"}
# note section headers that statistical NER sometimes labels as organisations
SECTION_HEADERS = {"review of systems", "history of present illness", "chief complaint", "physical exam",
                   "physical examination", "assessment and plan", "assessment", "plan", "medical history",
                   "past medical history", "social history", "family history", "surgical history", "past history",
                   "current medications", "medications", "allergies", "results", "vitals", "vitals reviewed",
                   "instructions", "impression", "hpi", "ros"}


@dataclass
class RecordContext:
    icd_tokens: set[str] = field(default_factory=set)
    icd_phrases: list[str] = field(default_factory=list)


class MedicalGuard:
    def __init__(self, lex: Lexicon):
        self.lex = lex

    def prepare(self, icd_codes: list[dict] | None) -> RecordContext:
        ctx = RecordContext()
        for item in icd_codes or []:
            desc = (item.get("description") or "").lower()
            ctx.icd_phrases.append(desc)
            for w in _WORD.findall(desc):
                ctx.icd_tokens.add(w.removesuffix("'s").strip("'"))
        return ctx

    def _eponym_in_medical_use(self, word: str, text: str, end: int, ctx: RecordContext, start: int = 0) -> str | None:
        parts = word.split("-")
        heads = self.lex.eponyms.get(word) or next((self.lex.eponyms[p] for p in parts if len(parts) > 1 and p in self.lex.eponyms),
                                                   None)  # "Dix-Hallpike": any part of a hyphenated eponym
        if heads is None:
            return None
        if "-" in word and word in self.lex.eponyms and \
                not re.search(r"\b(?:dr|mr|mrs|ms|miss|doctor|prof|professor)\.?\s*$", text[max(0, start - 14):start], re.I):
            return f"compound eponym '{word}'"  # "Dix-Hallpike", "Hawkins-Kennedy": never a hyphenated person name
        if re.search(r"\b(?:positive|negative|pos|neg|equivocal)\s+$", text[max(0, start - 12):start], re.I):
            return f"eponym '{word}' reported as a positive/negative examination finding"
        following = [w.lower().strip("'").removesuffix("'s") for w in _WORD.findall(text[end:end + 40])][:4]
        for f in following:
            if f in heads or f in MEDICAL_HEADS or f.removesuffix("s") in MEDICAL_HEADS:  # "Neer tests"
                return f"eponym '{word}' followed by medical head '{f}'"
            if f == "s" or f in self.lex.eponyms or f in ("and", "or"):  # possessive / double / coordinated eponyms
                continue
            break  # the head noun must follow immediately
        if re.match(r"(?:\s*(?:,|and|or)\s*[A-Z][\w'-]+)*\s+(?:tests?\s+)?(?:is|was|were|are)\s+(?:negative|positive|equivocal|"
                    r"normal|abnormal)\b", text[end:end + 80]):
            return f"eponym '{word}' reported as an examination finding"  # "Lachman and McMurray were negative"
        if word in ctx.icd_tokens and re.match(r"'s\b|s'", text[end:end + 2]):
            return f"possessive eponym '{word}' matches this record's ICD-10 description"
        return None

    def check(self, span: Span, text: str, ctx: RecordContext) -> str | None:
        """Return a reason string if the span is medical/system content that must be kept."""
        if span.label not in GUARDED_LABELS:
            return None
        surface = text[span.start:span.end]
        words = [w.lower().removesuffix("'s").strip("'") for w in _WORD.findall(surface)]
        words = [w for w in words if w not in {"mr", "mrs", "ms", "dr", "the", "of", "and"}]
        if not words:
            return None
        if all(w in SYSTEM_TERMS for w in words):
            return "voice-assistant / EHR system name, not a person or organisation"
        if " ".join(words) in SECTION_HEADERS:
            return "clinical note section header"
        # syntactic evidence (title, "my wife X", "works at X", "X hospital", "in X") outweighs
        # lexical medical evidence; only an eponym directly followed by its medical head overrides it
        strong_context = bool(span.meta.get("strong"))
        if not strong_context and len(words) >= 2 and words[-1].split("-")[-1] in INSTRUMENT_HEADS:  # also "DXA T-score"
            # "Epworth Sleepiness Scale 16/24", "Wells score", "Centor criteria": a named clinical instrument
            return f"named clinical instrument or test ('{words[-1]}')"
        if not strong_context:
            # "Tylenol Extra Strength", "Continue Xarelto", "ProAir inhaler": a drug/clinical head
            # accompanied only by ordinary (non-name) words is a medication mention
            heads = [w for w in words if self.lex.medical_strength(w) == 2 or w in ctx.icd_tokens]
            rest = [w for w in words if w not in heads and w not in DOSAGE_FORMS]
            if heads and all(self.lex.zipf(w) >= 3.5 and self.lex.name_ratio(w) < 0.1 for w in rest):
                return f"'{heads[0]}' is a known drug / clinical term (with ordinary words: {' '.join(rest) or '-'})"
            # "Renal Ultrasound Impression", "posterior chamber intraocular lens": a statistical ORG/LOC
            # whose words are mostly medical vocabulary is a clinical phrase, not an institution
            if span.label in ("ORGANIZATION", "LOCATION") and heads and \
                    sum(self.lex.medical_strength(w) >= 1 or w in ctx.icd_tokens for w in words) * 2 >= len(words):
                return f"mostly medical vocabulary ({', '.join(heads[:3])})"
        reasons = []
        for i, w in enumerate(words):
            ep = self._eponym_in_medical_use(w, text, span.end, ctx, span.start) if i == len(words) - 1 else None
            if ep:
                reasons.append(ep)
                continue
            if strong_context:
                return None
            if w in ctx.icd_tokens and self.lex.medical_strength(w) >= 1:
                reasons.append(f"'{w}' occurs in this record's ICD-10 description")
                continue
            strength = self.lex.medical_strength(w)
            if strength == 2:
                reasons.append(f"'{w}' is a known drug / clinical term")
                continue
            if span.label == "ORGANIZATION" and surface.isupper() and len(surface) <= 6 and w not in KNOWN_ORG_ACRONYMS:
                reasons.append(f"'{surface}' is an all-caps clinical abbreviation")
                continue
            if span.label in {"ORGANIZATION", "LOCATION"} and strength == 1 and self.lex.name_ratio(w) < 0.05:
                reasons.append(f"'{w}' is ICD-10 medical vocabulary")
                continue
            return None
        return "; ".join(reasons) if reasons else None
