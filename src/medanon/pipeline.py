"""End-to-end anonymisation of one encounter (dialogue + note), with an explanation per decision."""
from __future__ import annotations

import copy
import re
import sys
from collections import Counter

from .config import load_config
from .entities import EntityRegistry
from .lexicon import MEDICAL_HEADS, NOT_NAMES, RACE_ETHNICITY, SYSTEM_TERMS, WEEKDAYS, is_cased, load_lexicon
from .medical import MedicalGuard
from .ner import MLDetector, SpacyPresidioDetector, note_case_map
from .rules import _INST_RE, INSTITUTION_HEADS, RuleDetector
from .spans import RISK_CLASS, Span, resolve_overlaps
from .transform import Transformer, adult_band, age_band, year_in

# k-anonymity escalation ladder: policy overrides applied to records that are still unique
ESCALATION = [
    {},
    {"AGE": {"band_width": 20}, "OCCUPATION": {"action": "placeholder", "placeholder": "[OCCUPATION]"}},
    {"AGE": {"mode": "coarse"}, "OCCUPATION": {"action": "placeholder", "placeholder": "[OCCUPATION]"},
     "ETHNICITY": {"action": "placeholder", "placeholder": "[ETHNICITY]"}},
    # last resort: suppress age entirely (sex cannot be suppressed: it is carried by pronouns)
    {"AGE": {"action": "placeholder", "placeholder": "[AGE]"}, "OCCUPATION": {"action": "placeholder", "placeholder": "[OCCUPATION]"},
     "ETHNICITY": {"action": "placeholder", "placeholder": "[ETHNICITY]"}},
]
_DUP_PLACEHOLDER = re.compile(r"([\[<][A-Z_]+(?:_\d+)?[\]>])(?:[ \t]+\1)+")
# a word followed by a dose anywhere in the record is a medication ("ultram fifty milligrams")
_DOSE_AFTER = re.compile(r"\b([a-z][a-z\-]{3,})\s+(?:\d+(?:\.\d+)?|(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|"
                         r"ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
                         r"thousand|point|half|a|and)\s+)+)\s*(?:mg|milligrams?|mcg|micrograms?|grams?|units?|ml|puffs?|"
                         r"tablets?|capsules?)\b", re.I)
_SPEAKER_TAG = re.compile(r"\[[a-z_]+\]")
GENERIC_ROLES = {"doctor", "doctors", "nurse", "nurses", "physician", "patient", "patients", "provider", "therapist",
                 "surgeon", "specialist", "pcp", "dr", "practitioner", "assistant", "staff", "resident", "intern",
                 "attending", "caregiver", "guest", "dragon", "dax", "clinician", "clinicians", "person", "encounter",
                 "support", "the", "a", "an", "my", "your", "his", "her", "our", "their"}
_DIGIT_WORDS = re.compile(r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b", re.I)
# positions where a common word is used as a proper name: "at target", turn-initial "target has ..."
_ENTITY_SLOT = re.compile(r"(?:\b(?:at|for|from|with|to|in)\s+|^\[[a-z_]+\]\s+|^)$", re.I | re.M)
_WORD_CHAR = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9]")
# units of measurement ("mIU/L", "mg/dL", "mmHg", "x10^9/L") that statistical NER takes for organisations
_UNIT_RE = re.compile(r"(?:[mµμunpkcd]?(?:iu|u|g|mol|eq|l|m|osm)|mm\s?hg|bpm|cells|copies|x\s?10\^?\d+)"
                      r"(?:\s*/\s*(?:[mµμunpkcd]?(?:l|g|m)|kg|min|h|hrs?|day|d|wk|m2|1\.73\s?m2|hpf|mm3))*", re.I)
# a one-token "entity" the tagger reads as one of these parts of speech is not a name ("Reassess" -> VERB)
_NON_NAME_POS = {"VERB", "AUX", "ADV", "ADP", "PRON", "CCONJ", "SCONJ", "PART", "INTJ"}
_EDGE_PUNCT = ".,;:!? \t"
_CAMEL_ANALYTE = re.compile(r"(?=\S*[a-z][A-Z])[A-Za-z][A-Za-z0-9]{1,6}(?:-[A-Za-z0-9]{1,6})?")
_OCC_KEEP_WORDS = {"a", "an", "the", "my", "his", "her", "their", "our", "your", "as", "is", "was", "am", "are", "be", "been",
                   "i", "he", "she", "we", "they", "work", "works", "worked", "working", "job", "of", "at", "for", "in", "and",
                   "or", "retired", "former", "ex", "part-time", "full-time", "night", "nights", "day", "shift", "shifts",
                   "semi-retired", "self-employed", "unemployed", "current", "currently", "previous", "previously", "still",
                   "also", "now", "used", "to", "role", "main", "occupation", "activity", "employment", "do", "does"}
# analyte abbreviation with internal capitals (tTG-IgA, HbA1c, eGFR, pCO2) followed by a value
# a vital sign / bedside measurement with its value ("BP 114/71", "HR: 82", "SpO2 97%"): spaCy reads "BP" as a company
_VITAL_SPAN = re.compile(r"(?:B/?P|blood\s+pressure|HR|heart\s+rate|RR|resp(?:iratory)?\s+rate|PR|pulse|Sp[O0]2|Sa[O0]2|O2\s*sats?|"
                         r"sats?|temp(?:erature)?|T(?:max)?|wt|ht|BMI|GCS|MAP|FiO2|EF|LVEF)\b\.?\s*(?:[:=]|is|of|was)?\s*\d", re.I)
_BP_VALUE = re.compile(r"\b\d{2,3}\s*/\s*\d{2,3}\b")
# clinical section headers ("SOCIAL HX", "Review of Systems:"): structure, never an organisation or a place
_SECTION_WORDS = {"social", "family", "medical", "surgical", "past", "history", "hx", "exam", "examination", "physical",
                  "review", "of", "systems", "ros", "assessment", "plan", "impression", "medications", "meds", "allergies",
                  "vitals", "vital", "signs", "results", "labs", "laboratory", "data", "chief", "complaint", "cc", "hpi",
                  "pmh", "psh", "fhx", "shx", "subjective", "objective", "present", "illness", "reason", "for", "visit",
                  "diagnosis", "diagnoses", "procedures", "instructions", "follow", "up", "follow-up", "recommendations",
                  "problem", "list", "immunizations", "and", "&", "a/p", "current", "patient", "education", "orders"}
# the patient's care team named by organ or specialty ("heart doctor", "kidney doctors"): not the patient's job
# where the patient lives / comes from: a place named after such a cue is a residence even when its words are common
_RESIDENCE_CUE = re.compile(r"\b(?:lives?|living|reside[sd]?|moved|moving|relocated|grew\s+up|born|raised|commutes?|drive[sn]?|"
                            r"drove)\s+(?:(?:in|to|from|out|over|up|down|back)\s+){0,3}$", re.I)
_CARE_ROLES = {"doctor", "doctors", "specialist", "specialists", "surgeon", "surgeons", "nurse", "nurses", "team"}
_ANALYTE_VALUE = re.compile(r"(?=[A-Za-z0-9-]*[a-z][A-Z])[A-Za-z][A-Za-z0-9]{1,5}(?:-[A-Za-z0-9]{1,5})?\s+\d+(?:\.\d+)?")
_ABBREV_END = re.compile(r"\b(?:st|ave|rd|dr|blvd|ln|ct|hwy|inc|ltd|co|corp|jr|sr|mt|ft|apt|no)\.$", re.I)


_EXACT_WEIGHT_CONTEXT = re.compile(
    r"\b(?:mg|mcg|units?|ml)\s*/\s*kg\b|\bper\s+(?:kg|kilogram)\b|weight[- ]based|dry\s+weight|daily\s+weights?|"
    # pregnancy: only gestational weight-gain monitoring needs exact weights (a BMI class sets the targets);
    # an incidental mention ("no pregnancy plans") does not freeze body measurements
    r"\b(?:hemo|haemo|peritoneal\s+)?dialysis\b|fluid\s+overload|gestational\s+weight\s+gain|weight\s+gain\s+(?:in|during)\s+pregnancy|"
    r"chemotherapy|body\s+surface\s+area|\bmalnutrition\b|\bcachexia\b|failure\s+to\s+thrive|"
    r"anorexia\s+nervosa|\bbariatric\b", re.I)
_EXACT_WEIGHT_CODES = ("I50", "N18.5", "N18.6", "Z99.2", "E40", "E41", "E42", "E43", "E44", "E45", "E46", "R63.4",
                       "R63.5", "R64", "F50")


def _full_id_token(s: Span, text: str) -> Span:
    """A structured identifier is protected whole: "R4-MRN-[ID]" would keep the prefix of "R4-MRN-100103"."""
    if s.label != "ID":
        return s
    a, b = s.start, s.end
    while a > 1 and text[a - 1] in "-_/#" and text[a - 2].isalnum():
        a -= 1
        while a > 0 and text[a - 1].isalnum():
            a -= 1
    while b + 1 < len(text) and text[b] in "-_/#" and text[b + 1].isalnum():
        b += 1
        while b < len(text) and text[b].isalnum():
            b += 1
    return s if (a, b) == (s.start, s.end) else Span(s.field, a, b, s.label, s.source, s.score, text[a:b], dict(s.meta),
                                                       list(s.reasons))


def _trim_edges(s: Span, text: str) -> Span | None:
    """A detection never swallows the sentence punctuation around it ("18 Example Lane, East Lark." ->
    "[ADDRESS]."; "florist," -> "[OCCUPATION],"): the replacement keeps the sentence intact."""
    a, b = s.start, s.end
    while b > a and text[b - 1] in _EDGE_PUNCT:
        b -= 1
    while a < b and text[a] in _EDGE_PUNCT:
        a += 1
    if a >= b:
        return None
    if text[b:b + 1] == "." and _ABBREV_END.search(text[a:b + 1]) and re.match(r"\s+[a-z0-9]", text[b + 1:b + 3]):
        b += 1  # "12 Main St. in Springfield": the period belongs to the abbreviation, not to the sentence
    if (a, b) == (s.start, s.end):
        return s
    return Span(s.field, a, b, s.label, s.source, s.score, text[a:b], dict(s.meta), list(s.reasons))


def _device(requested: str | None) -> str:
    """'cpu' (default), 'cuda' or 'auto' (CUDA when available). Detection results do not depend on it."""
    if requested in (None, "", "cpu"):
        return "cpu"
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _split_on_newlines(spans: list[Span], texts: dict[str, str]) -> list[Span]:
    """Safety net: a replacement never removes a line break (each dialogue line is a speaker turn).
    A span that crosses lines is split into per-line parts that keep its label and evidence."""
    out = []
    for s in spans:
        text = texts[s.field]
        if "\n" not in text[s.start:s.end]:
            out.append(s)
            continue
        for m in re.finditer(r"[^\n]+", text[s.start:s.end]):
            if re.search(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9]", m.group(0)):
                a, b = s.start + m.start(), s.start + m.end()
                out.append(Span(s.field, a, b, s.label, s.source, s.score, text[a:b], dict(s.meta), list(s.reasons)))
    return out


class Anonymizer:
    def __init__(self, cfg: dict | None = None, config_path: str | None = None):
        self.cfg = cfg or load_config(config_path)
        self.lex = load_lexicon()
        self.rules = RuleDetector(self.lex, self.cfg)
        self.guard = MedicalGuard(self.lex)
        self.transformer = Transformer(self.cfg)
        sc = self.cfg["spacy"]
        self.spacy = SpacyPresidioDetector(self.lex, sc["model"], sc["presidio_patterns"], sc["truecase_dialogue"]) \
            if sc.get("enabled") else None
        mc = self.cfg["ml"]
        self.ml, self.ml_status = None, "disabled"
        if mc.get("enabled"):
            try:
                self.ml = MLDetector(mc["kind"], mc["model"], mc.get("threshold", 0.5),
                                     allow_download=mc.get("allow_download", True), device=_device(mc.get("device")))
                self.ml_status = "active"
            except Exception as exc:  # noqa: BLE001
                if mc.get("required", True):  # never silently run a different pipeline than configured
                    raise RuntimeError(f"required ML detector {mc['kind']}:{mc['model']} could not be loaded ({exc}). "
                                       f"Run `medanon prepare-models` once, or use the fast profile explicitly "
                                       f"(--no-ml / configs/fast.yaml).") from exc
                self.ml_status = f"unavailable ({exc}); running rules + spaCy only"
                print(f"[medanon] warning: ML detector {self.ml_status}", file=sys.stderr)
        # optional additional detectors (model experiments). A configured detector that cannot be
        # loaded is an error, never a silent skip.
        self.extra = []
        for ec in self.cfg.get("extra_detectors") or []:
            if ec.get("kind") == "llm":
                from .llm import LLMDetector
                det = LLMDetector(**{k: v for k, v in ec.items() if k not in ("kind", "mode", "fields")})
            else:
                det = MLDetector(ec["kind"], ec["model"], ec.get("threshold", 0.5),
                                 allow_download=ec.get("allow_download", True), label_map=ec.get("label_map"),
                                 device=_device(ec.get("device", mc.get("device"))))
            self.extra.append((ec, det))

    # ------------------------------------------------------------------ detection
    def detect(self, record: dict) -> tuple[dict, list[Span], EntityRegistry, list[dict]]:
        cfg = self.cfg
        eid = str(record.get("encounter_id", ""))
        texts = {f: record[f] for f in cfg["fields"] if isinstance(record.get(f), str)}
        ctx = self.guard.prepare(record.get("icd10_codes") if cfg["use_icd_context"] else None)
        case_map = note_case_map(texts.get("note", ""), self.lex)
        raw: list[Span] = []
        for fld, text in texts.items():
            found = self.rules.detect(text, fld)
            if self.spacy:
                found += self.spacy.detect(text, fld, case_map)
            if self.ml and fld in cfg["ml"].get("fields", texts):
                if cfg["ml"].get("mode", "cascade") == "cascade":
                    regions = self._uncertain_regions(text, found, ctx)
                    found += self.ml.detect(text, fld, regions) if regions else []
                else:
                    found += self.ml.detect(text, fld)
            for ec, det in self.extra:
                if fld not in ec.get("fields", ["dialogue", "note"]):
                    continue
                if ec.get("mode", "full") == "cascade":
                    regions = self._uncertain_regions(text, found, ctx)
                    found += det.detect(text, fld, regions) if regions else []
                else:
                    found += det.detect(text, fld)
            raw += found

        kept: list[dict] = []
        spans: list[Span] = []
        # structural guard: speaker tags ([doctor], [patient], [patient_guest]) are format, never PHI
        tags = {fld: [(m.start(), m.end()) for m in _SPEAKER_TAG.finditer(t)] for fld, t in texts.items()}
        raw = [s for s in raw if not any(a < s.end and s.start < b for a, b in tags[s.field])]
        med_terms = {m.group(1).lower() for t in texts.values() for m in _DOSE_AFTER.finditer(t)}
        for s in raw:
            statistical = not s.source.startswith(("rule", "presidio"))
            if statistical and s.score < cfg["min_score"].get(s.label, 0.5):
                continue
            if statistical and s.label in ("ORGANIZATION", "LOCATION") and \
                    texts[s.field][s.start:s.end].strip().lower() in med_terms:
                continue  # dosed somewhere in this record: a medication, not an organisation or place
            code = re.fullmatch(r"([A-Za-z]{1,6})-\d{1,3}", texts[s.field][s.start:s.end].strip())
            if statistical and s.label in ("ORGANIZATION", "LOCATION") and code and \
                    self.lex.medical_strength(code.group(1)) == 2:
                continue  # clinical code "chem-12", "b-12": a test or analyte, not an organisation
            if statistical:
                s = self._clean_statistical(s, texts[s.field])
                if s is None:
                    continue
            s = _trim_edges(s, texts[s.field])
            if s is None or (s.label == "OCCUPATION" and s.text.isupper() and len(s.text) > 3):
                continue  # an all-caps note header ("HISTORY") is not an occupation
            reason = self.guard.check(s, texts[s.field], ctx)
            if reason is None and s.label == "PERSON" and statistical:
                words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]+", s.text)]
                if words and all(self.lex.name_ratio(w) < 0.1 and self.lex.zipf(w) >= 4.0 for w in words):
                    reason = "statistical PERSON guess on common word(s) without name evidence"
            if reason:
                kept.append(self._keep_decision(eid, s, reason))
                continue
            spans.append(s)

        registry = EntityRegistry(self.lex, cfg["role_aware_placeholders"])
        persons = [s for s in spans if s.label == "PERSON"]
        persons.sort(key=lambda s: (s.field != "note", -len(s.meta.get("name_tokens") or s.text.split()), s.start))
        for s in persons:
            registry.add(s, texts)
        registry.merge_overlapping()
        registry.finalize()
        for s in registry.propagate(texts) + self._propagate_places_orgs(spans, texts) + \
                self._propagate_occupations(spans, texts):
            s = _trim_edges(s, texts[s.field])
            if s is not None and self.guard.check(s, texts[s.field], ctx) is None:
                spans.append(s)
        def protected(fld: str, a: int, b: int) -> bool:
            return any(self.lex.medical_strength(w) == 2 for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]+", texts[fld][a:b]))

        final = _split_on_newlines(resolve_overlaps([_full_id_token(s, texts[s.field]) for s in spans], protected), texts)
        self._mark_exact_measurements(final, texts, record)
        return texts, final, registry, kept

    def _mark_exact_measurements(self, spans: list[Span], texts: dict[str, str], record: dict) -> None:
        """Body measurements stay exact where care depends on them: children (growth, dosing), weight-based
        dosing, dialysis / fluid balance, pregnancy, malnutrition; BMI also when an ICD-10 BMI code (Z68) is
        coded. Otherwise they are generalised (transform.anthro_interval)."""
        anthro = [s for s in spans if s.label == "ANTHROPOMETRIC"]
        if not anthro:
            return
        codes = [c.get("code", "") for c in record.get("icd10_codes") or []]
        age = self._patient_age(spans)
        reason = None
        if age is not None and age < 18:
            reason = "paediatric patient (growth assessment, weight-based dosing)"
        elif m := _EXACT_WEIGHT_CONTEXT.search(" ".join(texts.values())):
            reason = f"clinical context '{m.group(0).lower()}' needs the exact value"
        elif any(c.startswith(_EXACT_WEIGHT_CODES) for c in codes):
            reason = "ICD-10 code of the record depends on the exact measurement"
        for s in anthro:
            if reason or (s.meta.get("kind") == "bmi" and any(c.startswith("Z68") for c in codes)):
                s.meta["exact_reason"] = reason or "ICD-10 BMI code (Z68) coded for this encounter"

    # ------------------------------------------------------------------ helpers
    def _uncertain_regions(self, text: str, found: list[Span], ctx) -> list[tuple[int, int]]:
        """Cascade trigger: lines containing a token that could be an identifier but that no cheap
        detector explained - a name-lexicon token or a rare non-dictionary word that is not medical."""
        covered = set()
        for s in found:
            covered.update(range(s.start, s.end))
        regions = []
        for line in re.finditer(r"[^\n]+", text):
            for m in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]{2,}", line.group(0)):
                w = m.group(0).lower().removesuffix("'s")
                pos = line.start() + m.start()
                if pos in covered or w in NOT_NAMES or w in SYSTEM_TERMS or w in ctx.icd_tokens:
                    continue
                if self.lex.medical_strength(w) or w in self.lex.eponyms:
                    continue
                if self.lex.name_ratio(w) >= 0.05 or (self.lex.zipf(w) < 3.0 and w not in self.lex.icd_vocab):
                    regions.append((line.start(), line.end()))
                    break
        return regions

    def _clean_statistical(self, s: Span, text: str) -> Span | None:
        """Sanity-filter statistical NER spans before they are trusted.

        * never cross a line break (section headers / lab blocks are not one entity)
        * PERSON: drop leading/trailing common non-name words ("Continue Xarelto" -> "Xarelto",
          which the medical guard then recognises as a drug)
        * NATIONALITY: languages and foods ("speaks English", "french fries") are not identifiers
        """
        surf = text[s.start:s.end]
        a, b = s.start + len(surf) - len(surf.lstrip()), s.end - len(surf) + len(surf.rstrip())
        if a >= b:
            return None
        if (a, b) != (s.start, s.end):  # sentencepiece models (OpenMed/DeBERTa) include the leading space
            s = Span(s.field, a, b, s.label, s.source, s.score, text[a:b], dict(s.meta), list(s.reasons))
        nl = text.find("\n", s.start, s.end)
        if nl != -1:
            if nl == s.start:
                return None
            s = Span(s.field, s.start, nl, s.label, s.source, s.score, text[s.start:nl], dict(s.meta), list(s.reasons))
        # never cut through a word ("SpO" of "SpO2", "Joh" of "Johnson"): the guard judges the whole token
        a, b = s.start, s.end
        while a > 0 and _WORD_CHAR.match(text[a - 1]):
            a -= 1
        while b < len(text) and _WORD_CHAR.match(text[b]):
            b += 1
        # ... nor through a hyphenated compound: "Kennedy" of "Hawkins-Kennedy" is judged as the whole eponym
        while a > 1 and text[a - 1] == "-" and text[a - 2].isalpha():
            a -= 1
            while a > 0 and _WORD_CHAR.match(text[a - 1]):
                a -= 1
        while b + 1 < len(text) and text[b] == "-" and text[b + 1].isalpha():
            b += 1
            while b < len(text) and _WORD_CHAR.match(text[b]):
                b += 1
        if (a, b) != (s.start, s.end):
            s = Span(s.field, a, b, s.label, s.source, s.score, text[a:b], dict(s.meta), list(s.reasons))
        if s.label not in ("ID", "CONTACT") and self._is_unit(s, text):
            return None  # "TSH 8.1 mIU/L", "the letters in mIU/L": a unit of measurement, not a name
        m = re.match(r"(\S+?)[.,;:]\s+(?=\S)", s.text)
        if m and _UNIT_RE.fullmatch(m.group(1)):  # "ng/dL. Family": a unit glued to the next sentence
            s = Span(s.field, s.start + m.end(), s.end, s.label, s.source, s.score, s.text[m.end():], dict(s.meta),
                     list(s.reasons))
        tok = s.text.strip(" .,;:")
        if s.label in ("ORGANIZATION", "PERSON", "LOCATION", "NATIONALITY"):
            line_start = text.rfind("\n", 0, s.start) + 1
            line_end = text.find("\n", s.end)
            line = text[line_start:len(text) if line_end == -1 else line_end]
            if _VITAL_SPAN.match(s.text) or _BP_VALUE.search(s.text) or (
                    _VITAL_SPAN.match(text[s.start:s.start + 40]) and not re.search(r"[a-z]{3,}", s.text)):
                return None  # "BP 114/71": a vital sign with its value, not an organisation
            words_h = re.findall(r"[A-Za-z/&\-]+", line)
            if words_h and len(line.strip(" :\t")) <= 40 and line.strip(" :\t").lower().startswith(s.text.strip(" :").lower()[:3]) \
                    and all(w.lower() in _SECTION_WORDS for w in words_h):
                return None  # "SOCIAL HX", "Review of Systems:": a section header
            words_c = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", s.text)]
            if words_c and all(self.lex.medical_strength(w.removesuffix("'s")) >= 1 and self.lex.name_ratio(w) < 0.05
                               for w in words_c):
                return None  # "lungs", "neurovascular's": clinical vocabulary, never a name, place or organisation
        if s.label == "OCCUPATION":
            words_o = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", s.text)]
            if len(words_o) >= 2 and words_o[-1] in _CARE_ROLES and all(
                    self.lex.medical_strength(w) >= 1 or w in ("the", "my", "his", "her", "a", "an") for w in words_o[:-1]):
                return None  # "heart doctor", "my kidney doctors": the patient's care team, not their occupation
        if s.label in ("ORGANIZATION", "PERSON", "LOCATION") and _CAMEL_ANALYTE.fullmatch(tok) \
                and sum(c.isupper() for c in tok) >= 2 and not self.lex.in_name_lexicon(tok.lower()):
            return None  # "tTG-IgA", "IgA", "eGFR": a mixed-case analyte abbreviation (surnames such as McDonald excluded)
        if s.label not in ("ID", "CONTACT") and _ANALYTE_VALUE.fullmatch(s.text.strip(" .,;:")):
            return None  # "tTG-IgA 68", "HbA1c 7.2": a mixed-case lab analyte with its value, not an organisation
        if s.label in ("ORGANIZATION", "LOCATION") and s.text.islower() and is_cased(text):
            return None  # cased text: an organisation / place name is capitalised ("recheck labs" is not one)
        if s.source.startswith("spacy") and s.label in ("ORGANIZATION", "LOCATION"):
            pos = [p for p in s.meta.get("pos", []) if p not in ("DET", "PUNCT")]
            if len(pos) == 1 and pos[0] in _NON_NAME_POS:
                return None  # "Reassess for imaging": the tagger reads a verb where NER guessed an organisation
        if s.label in ("ORGANIZATION", "LOCATION") and not s.meta.get("org_type"):
            name = s.text.rstrip(".,;:!? ")
            heads = list(_INST_RE.finditer(name))
            if heads and heads[-1].start() > 0 and heads[-1].end() == len(name) and \
                    INSTITUTION_HEADS[heads[-1].group(0).lower()] != "location":
                # "Pinebrook Community Clinic" (spaCy FAC -> LOCATION) is a named healthcare organisation
                s = Span(s.field, s.start, s.end, "ORGANIZATION", s.source, s.score, s.text,
                         {**s.meta, "org_type": INSTITUTION_HEADS[heads[-1].group(0).lower()]}, list(s.reasons))
        if s.source.startswith("ml:"):
            # transformer spans are accepted only with lexical support: occupations must not be generic
            # clinical roles or medical terms (models tag "cardiologist", "the nurse", even "[doctor]"),
            # people need name evidence, places / organisations a distinctive, non-medical word
            words = [w.lower().removesuffix("'s") for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9'\-]*", text[s.start:s.end])]
            if not words and s.label not in ("ID", "CONTACT"):
                return None
            # the encounter's own staff ("the nurse", "clinician", "the cardiologist") is not the patient's job;
            # a qualified occupation ("laboratory assistant", "retired nurse") is
            if s.label == "OCCUPATION" and words and (all(w in GENERIC_ROLES for w in words)
                                                      or self.lex.medical_strength(words[-1]) == 2
                                                      or all(w in NOT_NAMES for w in words)):
                return None
            if s.label == "OCCUPATION":
                # job-specific qualifiers belong to the title ("former city bus mechanic" -> "former [OCCUPATION]");
                # status, schedule and exposure words stay readable (clinically relevant, not identifying)
                a = s.start
                for _ in range(3):
                    m = re.search(r"([a-z][a-z\-]+)\s+$", text[max(0, a - 30):a])
                    if not m or m.group(1) in _OCC_KEEP_WORDS or self.lex.medical_strength(m.group(1)) == 2:
                        break
                    a -= len(m.group(0))
                if a != s.start:
                    s = Span(s.field, a, s.end, s.label, s.source, s.score, text[a:s.end], dict(s.meta), list(s.reasons))
                    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", s.text)]
            if s.label == "OCCUPATION" and not s.meta.get("category"):
                # same coarse category as a rule-detected occupation ("florist" -> its lexicon category)
                cat = next((c for w in reversed(words) for c in (self.lex.occupations.get(w),
                                                                 self.lex.occupations.get(w.removesuffix("s"))) if c), None)
                if cat:
                    s = Span(s.field, s.start, s.end, s.label, s.source, s.score, s.text, {**s.meta, "category": cat},
                             list(s.reasons))
            if s.label in ("ID", "CONTACT") and not re.search(r"[\d@]", text[s.start:s.end]) \
                    and len(_DIGIT_WORDS.findall(text[s.start:s.end])) < 3:
                return None  # "email", "a1c" are words, not identifiers (spoken digit runs are kept)
            if any(self.lex.medical_strength(w) == 2 for w in words) and s.label != "PERSON":
                return None
            if s.label == "PERSON" and not any(
                    self.lex.name_ratio(w) >= 0.05 or (self.lex.zipf(w) < 3.0 and self.lex.medical_strength(w) == 0)
                    for w in words if w not in NOT_NAMES):
                return None
            if s.label in ("LOCATION", "ORGANIZATION") and not any(
                    (self.lex.zipf(w) < 4.0 or w in self.lex.places or self.lex.name_ratio(w) >= 0.1)
                    and self.lex.medical_strength(w) == 0 and w not in NOT_NAMES for w in words):
                return None
        if s.label in ("ORGANIZATION", "LOCATION") and s.source.startswith("spacy"):
            # a statistical ORG/LOC made only of ordinary English words ("west coast", "micro",
            # "urgent care provider") is not identifying; real names contain a rare/proper word
            words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]*", text[s.start:s.end])]
            if words and all(self.lex.zipf(w) >= 4.0 and self.lex.name_ratio(w) < 0.1 and w not in self.lex.places
                             for w in words) and not (s.label == "LOCATION" and _RESIDENCE_CUE.search(
                                 text[max(0, s.start - 30):s.start])):
                return None
        if s.label == "NATIONALITY" and not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ\- ]{3,}", text[s.start:s.end]):
            return None  # "V8", "d." - tokenisation debris, not a nationality
        if s.label == "NATIONALITY" and text[s.start:s.end].lower() in RACE_ETHNICITY:
            s = Span(s.field, s.start, s.end, "ETHNICITY", s.source, s.score, s.text, dict(s.meta), list(s.reasons))
        if s.label == "NATIONALITY":
            after = text[s.end:s.end + 25].lower()
            before = text[max(0, s.start - 25):s.start].lower()
            if re.match(r"\s*(?:fries|toast|bread|food|cuisine|dressing|press|onion|vanilla|horn|bulldog|language)", after) \
                    or re.search(r"(?:speaks?|speaking|in|interpreter|language|fluent)\s*$", before) \
                    or re.match(r"\s*(?:-?speaking|speaker|interpreter)", after):
                return None
        if s.label != "PERSON":
            return s
        words = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9'\-]*", text[s.start:s.end]))
        def common(w: str) -> bool:
            w = w.lower().removesuffix("'s")
            return self.lex.name_ratio(w) < 0.1 and self.lex.zipf(w) >= 4.0
        while words and common(words[0].group(0)):
            words.pop(0)
        while words and common(words[-1].group(0)):
            last, prev = words[-1].group(0), words[-2].group(0) if len(words) > 1 else ""
            if last[:1].isupper() and prev[:1].isupper() and self.lex.name_ratio(prev.lower()) >= 0.3 \
                    and self.lex.medical_strength(last) == 0:
                break  # "Michelle King": a capitalised word after a first name is a surname, even if common
            words.pop()
        if words and re.match(r"\s+na\b", text[s.start + words[-1].end():]) and \
                self.lex.zipf(words[-1].group(0).lower() + "na") >= 4.0:
            return None  # ASR split contraction ("wan na", "gon na"), not a name
        if not words:
            return None
        start, end = s.start + words[0].start(), s.start + words[-1].end()
        if (start, end) != (s.start, s.end):
            s = Span(s.field, start, end, s.label, s.source, s.score, text[start:end], dict(s.meta), list(s.reasons))
        if self._is_unit(s, text):
            return None  # trimming can leave only a unit ("T4 0.9 ng" -> "ng")
        # cased text: a first name followed by a capitalised census surname is one person ("Lark March"),
        # even when the surname is also an ordinary word - unless it starts a date or an eponymous term
        m = re.match(r" ([A-Z][a-z'\-]+)\b", text[s.end:s.end + 30])
        if m and len(s.text.split()) == 1 and is_cased(text) and m.group(1).lower() in self.lex.surnames \
                and m.group(1).lower() not in WEEKDAYS and not re.match(r"\s*\d", text[s.end + m.end():s.end + m.end() + 4]) \
                and not any(w.lower() in MEDICAL_HEADS for w in re.findall(r"[A-Za-z]+", text[s.end + m.end():s.end + m.end() + 15])[:1]):
            s = Span(s.field, s.start, s.end + m.end(), s.label, s.source, s.score, text[s.start:s.end + m.end()],
                     dict(s.meta), list(s.reasons))
        return s

    @staticmethod
    def _is_unit(s: Span, text: str) -> bool:
        surf = s.text.strip(" .,;:")
        return bool(_UNIT_RE.fullmatch(surf)) and ("/" in surf or bool(re.search(r"\d\s*$", text[max(0, s.start - 12):s.start])))

    def _propagate_occupations(self, spans: list[Span], texts: dict[str, str]) -> list[Span]:
        """An occupation detected in one field is the same disclosure in the other ("repair bicycles" in
        the note and in the dialogue): transform every mention, so the policy is applied consistently."""
        out, seen = [], set()
        for s in spans:
            if s.label != "OCCUPATION":
                continue
            surface = texts[s.field][s.start:s.end].strip()
            words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]*", surface)
            if surface.lower() in seen or not words or (len(words) == 1 and self.lex.zipf(words[0]) >= 4.5):
                continue  # a very common single word is only transformed where it was detected
            seen.add(surface.lower())
            rx = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9])" + r"\s+".join(map(re.escape, surface.split())) +
                            r"(?![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9])", re.I)
            for fld, text in texts.items():
                for m in rx.finditer(text):
                    out.append(Span(fld, m.start(), m.end(), "OCCUPATION", "propagation", 0.8, m.group(0),
                                    meta={k: v for k, v in s.meta.items() if k == "category"},
                                    reasons=["same occupation as detected elsewhere in this encounter"]))
        return out

    def _propagate_places_orgs(self, spans: list[Span], texts: dict[str, str]) -> list[Span]:
        """A location / organisation identified anywhere in the encounter is identifying everywhere:
        search its other mentions in both fields (case-insensitive; capitalised in cased text)."""
        out, seen = [], set()
        for s in spans:
            if s.label not in {"LOCATION", "ORGANIZATION"}:
                continue
            surface = re.sub(r"^(?:the|a|an)\s+", "", texts[s.field][s.start:s.end].strip(), flags=re.I)
            key = surface.lower()
            words = re.findall(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ][a-zß-öø-ÿĀ-ɏḀ-ỿ'\-]*", key)
            if key in seen or not words:
                continue
            seen.add(key)
            distinctive = len(words) > 1 or (self.lex.zipf(words[0]) < 3.5 and self.lex.medical_strength(words[0]) < 2)
            if not distinctive and not s.meta.get("strong"):
                continue  # a common word (e.g. "target") is only propagated from a syntactically anchored mention
            rx = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9])" + r"\s+".join(map(re.escape, surface.split())) + r"(?![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9])", re.I)
            for fld, text in texts.items():
                cased = fld == "note"
                for m in rx.finditer(text):
                    if cased and not m.group(0)[:1].isupper():
                        continue
                    if not distinctive and not _ENTITY_SLOT.search(text[max(0, m.start() - 25):m.start()]):
                        continue
                    out.append(Span(fld, m.start(), m.end(), s.label, "propagation", 0.8, m.group(0),
                                    meta={k: v for k, v in s.meta.items() if k in ("org_type", "place_type")},
                                    reasons=[f"same {s.label.lower()} as detected elsewhere in this encounter"]))
        return out

    # ------------------------------------------------------------------ transformation
    def anonymize(self, record: dict, include_text: bool = False) -> tuple[dict, list[dict]]:
        """Anonymise one encounter independently (no dataset-level k-anonymity escalation)."""
        return self.render(record, self.detect(record), include_text=include_text)

    def render(self, record: dict, detection, include_text: bool = False, level: int = 0,
               escalation_reason: str = "") -> tuple[dict, list[dict]]:
        eid = str(record.get("encounter_id", ""))
        texts, spans, registry, kept = detection
        decisions = list(kept)
        overrides = self._overrides(level, spans)
        out = copy.deepcopy(record)
        for fld, text in texts.items():
            pieces, pos = [], 0
            for s in sorted((x for x in spans if x.field == fld), key=lambda x: x.start):
                ent = registry.entity_for(s, texts) if s.label == "PERSON" else None
                rep, action, expl = self.transformer.apply(s, text, eid, ent.placeholder if ent else None, overrides)
                surface = text[s.start:s.end]
                fmt = self.cfg.get("placeholder_format", "[{}]")
                if rep and fmt != "[{}]" and rep.startswith("[") and rep.endswith("]"):
                    rep = fmt.format(rep[1:-1])
                if rep and s.label == "PERSON" and re.search(r"'s$", surface, re.I):
                    rep += "'s"
                if level and s.label in overrides and action != "keep":
                    expl += f" | k-anonymity escalation level {level}: {escalation_reason}"
                dec = {
                    "encounter_id": eid, "field": fld, "start": s.start, "end": s.end, "label": s.label,
                    "action": action, "replacement": rep, "risk": RISK_CLASS.get(s.label, ""),
                    "detectors": s.meta.get("sources", [s.source]), "confidence": round(s.score, 2),
                    "explanation": expl + " | evidence: " + "; ".join(s.reasons[:4]),
                }
                if ent:
                    dec["role"] = ent.role
                if action == "keep":
                    dec["reason_code"] = "POLICY_KEEP"  # detected, deliberately published (exact age, calendar year)
                decisions.append(dec)
                if rep is None:
                    continue
                pieces.append(text[pos:s.start])
                pieces.append(rep)
                pos = s.end
            pieces.append(text[pos:])
            out[fld] = _DUP_PLACEHOLDER.sub(r"\1", "".join(pieces))
        # audit privacy: original surfaces only on explicit request; otherwise also scrub them from
        # explanations (guard reasons quote the word they judged, e.g. "eponym 'bell' ...")
        for d in decisions:
            surface = texts[d["field"]][d["start"]:d["end"]]
            if include_text:
                d["original"] = surface
            else:
                d["explanation"] = _redact(d["explanation"], surface)
        return out, decisions

    # ------------------------------------------------------------------ dataset level (k-anonymity)
    def quasi_identifiers(self, detection, level: int) -> tuple:
        """Published demographic quasi-identifiers of a record after transformation at `level`."""
        texts, spans, _, _ = detection
        ages = [s.meta.get("age") for s in sorted(spans, key=lambda s: (s.field != "note", s.start))
                if s.label == "AGE" and s.meta.get("age") is not None]
        age_pol = {**self.cfg["policies"]["AGE"], **self._overrides(level, spans).get("AGE", {})}
        top = age_pol.get("top_code", 90)
        if not ages or age_pol.get("action") not in ("keep", "generalize", "exact", "band"):
            age = None
        elif age_pol.get("action") == "exact":  # what is published: the exact age, or "90+"
            age = f"{top}+" if ages[0] >= top else str(int(ages[0]))
        elif age_pol.get("action") == "band":  # exact paediatric age, adult band, "90+"
            age = adult_band(ages[0], age_pol.get("band_width", 10), top, age_pol.get("exact_below", 18)) or str(int(ages[0]))
        else:
            age = age_band(ages[0], age_pol.get("band_width", 10), top, age_pol.get("mode", "band"))
        note = texts.get("note", "") + " " + texts.get("dialogue", "")[:3000]
        male = len(re.findall(r"\b(?:he|his|him|male|man|gentleman|mr)\b", note, re.I))
        female = len(re.findall(r"\b(?:she|her|hers|female|woman|lady|mrs|ms)\b", note, re.I))
        sex = "M" if male > 1.5 * female else "F" if female > 1.5 * male else "U"
        occ = sorted({s.meta.get("category") or "other" for s in spans if s.label == "OCCUPATION"})  # as in transform
        eth = sorted({s.text.lower() for s in spans if s.label == "ETHNICITY"})
        occ_pub = tuple(occ) if self._published("OCCUPATION", level) else ()
        eth_pub = tuple(eth) if self._published("ETHNICITY", level) else ()
        return age, sex, occ_pub, eth_pub

    def _overrides(self, level: int, spans) -> dict:
        """Escalation overrides for this record. An exact-age policy is an explicit team decision: the QI step
        never widens or suppresses the age. A birth year is never kept when the patient is 90 or older."""
        ov = copy.deepcopy(ESCALATION[level])
        age_pol = self.cfg["policies"].get("AGE", {})
        if age_pol.get("action") in ("exact", "band"):
            ov.pop("AGE", None)
        if any(s.label == "AGE" and (s.meta.get("age") or 0) >= age_pol.get("top_code", 90) for s in spans):
            ov.setdefault("DOB", {})["keep_year"] = False
        if age_pol.get("action") == "band" and self._patient_age(spans) is not None and \
                self._patient_age(spans) >= age_pol.get("exact_below", 18):
            # a published birth year would give the exact adult age back (visit year - birth year)
            ov.setdefault("DOB", {})["keep_year"] = False
        return ov

    def _patient_age(self, spans) -> int | None:
        """The patient's age as stated (note first), else from the birth year (reference year of the DOB policy)."""
        ages = [s.meta["age"] for s in sorted(spans, key=lambda s: (s.field != "note", s.start))
                if s.label == "AGE" and s.meta.get("age") is not None]
        if ages:
            return int(ages[0])
        years = [y[1] for s in spans if s.label == "DOB" and (y := year_in(s.text)) and y[1]]
        ref = self.cfg["policies"].get("DOB", {}).get("reference_year", 2026)
        return ref - years[0] if years else None

    def _published(self, label: str, level: int) -> bool:
        pol = {**self.cfg["policies"].get(label, {}), **ESCALATION[level].get(label, {})}
        return pol.get("action") in ("keep", "generalize")

    def anonymize_dataset(self, records: list[dict], include_text: bool = False, detections: list | None = None):
        """Two-pass anonymisation: detect all records, then escalate generalisation for records whose
        published quasi-identifier combination is shared by fewer than k records (k-anonymity)."""
        kcfg = self.cfg.get("k_anonymity", {})
        detections = detections or [self.detect(r) for r in records]
        levels = [0] * len(records)
        reasons = [""] * len(records)
        if kcfg.get("enabled", True) and len(records) > 1:
            k = kcfg.get("k", 3)
            for level in range(1, len(ESCALATION)):
                sigs = [self.quasi_identifiers(d, lv) for d, lv in zip(detections, levels)]
                counts = Counter(sigs)
                for i, sig in enumerate(sigs):
                    if counts[sig] < k and levels[i] == level - 1 and any(sig[:1] + sig[2:]):
                        levels[i] = level
                        reasons[i] = (f"published quasi-identifiers (age, sex, occupation, ethnicity) shared by "
                                      f"only {counts[sig]} record(s) in this dataset (< k={k})")
        outputs = [self.render(r, d, include_text, lv, why) for r, d, lv, why in zip(records, detections, levels, reasons)]
        stats = {"escalation_levels": {str(k): v for k, v in sorted(Counter(levels).items())}}
        # post-hoc verification on what is actually published (declared QI set only)
        k = kcfg.get("k", 3)
        final = Counter(self.quasi_identifiers(d, lv) for d, lv in zip(detections, levels))
        sizes = [final[self.quasi_identifiers(d, lv)] for d, lv in zip(detections, levels)]
        stats["qi_verification"] = {
            "qi_set": ("exact age (up to 89, policy)" if self.cfg["policies"].get("AGE", {}).get("action") == "exact"
                       else "age band") + ", sex (inferred from pronouns), occupation category, ethnicity", "k": k,
            "min_class_size": min(sizes) if sizes else 0, "records_below_k": sum(s < k for s in sizes),
            "scope": "heuristic generalisation of declared demographic quasi-identifiers; free text, dates, "
                     "diagnoses (ICD codes) and lab values are NOT covered - this is not formal k-anonymity"}
        return outputs, stats

    @staticmethod
    def _keep_decision(eid: str, s: Span, reason: str) -> dict:
        # no surface text here: render() adds it only when include_text is requested
        return {"encounter_id": eid, "field": s.field, "start": s.start, "end": s.end, "label": "MEDICAL_CONTEXT",
                "candidate_label": s.label, "action": "keep", "replacement": None, "detectors": [s.source],
                "confidence": round(s.score, 2), "reason_code": _reason_code(reason),
                "explanation": f"kept: candidate {s.label} judged clinical/system content - {reason}"}


def _reason_code(reason: str) -> str:
    for key, code in (("eponym", "EPONYM_IN_MEDICAL_USE"), ("drug", "DRUG_OR_CLINICAL_TERM"),
                      ("ICD-10 description", "RECORD_ICD10_TERM"), ("abbreviation", "CLINICAL_ABBREVIATION"),
                      ("section header", "SECTION_HEADER"), ("system", "SYSTEM_NAME"),
                      ("mostly medical", "MEDICAL_PHRASE"), ("ICD-10 medical vocabulary", "MEDICAL_VOCABULARY"),
                      ("common word", "COMMON_WORD_NOT_NAME")):
        if key in reason:
            return code
    return "OTHER"


def _redact(text: str, surface: str) -> str:
    """Remove every word of the original surface from an explanation string."""
    for w in sorted(set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9'\-]*", surface)), key=len, reverse=True):
        if len(w) >= 2:
            text = re.sub(rf"(?<![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9]){re.escape(w)}(?![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9])", "<span>", text, flags=re.I)
    return text
