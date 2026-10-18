"""Offline lexical resources and small language utilities.

Everything here is loaded from ``medanon/resources`` (built by ``scripts/build_resources.py``
from public-domain sources); nothing touches the network at inference time.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RES_DIR = Path(__file__).parent / "resources"

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
FULL_MONTHS = [m for m in MONTHS if len(m) > 4 or m in ("may", "june", "july")]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
HOLIDAYS = [
    "christmas eve", "christmas", "thanksgiving", "easter", "halloween", "new year's eve", "new year's day",
    "new years eve", "new years", "new year's", "fourth of july", "4th of july", "independence day",
    "memorial day", "labor day", "valentine's day", "valentines day", "mother's day", "father's day",
    "hanukkah", "chanukah", "ramadan", "passover", "yom kippur", "rosh hashanah", "diwali", "kwanzaa",
    "super bowl", "spring break", "black friday", "veterans day", "presidents day", "st. patrick's day",
]
TITLES = {"mr", "mrs", "ms", "miss", "mister", "missus", "dr", "doctor", "doc", "nurse", "prof", "professor"}
CLINICIAN_TITLES = {"dr", "doctor", "doc", "nurse", "prof", "professor"}
RELATIONS = {
    "wife", "husband", "son", "daughter", "mother", "father", "mom", "mum", "mommy", "dad", "daddy",
    "sister", "brother", "partner", "spouse", "fiance", "fiancee", "fiancé", "girlfriend", "boyfriend",
    "friend", "neighbor", "neighbour", "boss", "coworker", "co-worker", "colleague", "cousin", "aunt",
    "uncle", "niece", "nephew", "grandson", "granddaughter", "grandmother", "grandfather", "grandma",
    "grandpa", "granny", "stepson", "stepdaughter", "stepmother", "stepfather", "son-in-law",
    "daughter-in-law", "roommate", "caregiver", "kid", "kids", "child", "baby", "twin", "sibling",
}
PET_WORDS = {"dog", "cat", "puppy", "kitten", "horse", "pup", "pet", "parrot", "bird"}
POSSESSIVES = {"my", "his", "her", "your", "their", "our"}
GREETINGS = {"hi", "hello", "hey", "morning", "afternoon", "evening", "thanks", "bye", "goodbye", "welcome"}
# words that look like names in some lexicon but must never be treated as personal names here
NOT_NAMES = {
    "sir", "ma'am", "maam", "madam", "doc", "doctor", "nurse", "dear", "honey", "sweetie", "sweetheart",
    "guys", "everyone", "everybody", "folks", "man", "dude", "babe", "baby", "kiddo", "champ",
    "dragon", "okay", "ok", "yes", "yeah", "yep", "no", "nope", "well", "so", "um", "uh", "hmm", "mm",
    "mhmm", "alright", "today", "tomorrow", "yesterday", "tonight", "again", "too", "also", "there",
    "here", "now", "then", "please", "thanks", "thank", "sure", "great", "good", "fine", "right", "bye",
    "patient", "mother", "father", "mom", "dad", "son", "daughter", "wife", "husband", "brother",
    "sister", "god", "lord", "jesus", "christ", "male", "female", "gentleman", "lady", "pain", "and",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "is", "are", "was", "be", "it", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "this", "that", "who", "what",
    "how", "why", "when", "where", "which", "let", "lets", "let's", "just", "very", "much", "more",
}
SYSTEM_TERMS = {"dragon", "epic", "cerner", "mychart", "nuance", "dax", "athena", "allscripts", "meditech"}
RACE_ETHNICITY = {
    "white", "black", "caucasian", "african american", "african-american", "hispanic", "latino", "latina",
    "latinx", "asian", "asian american", "native american", "pacific islander", "alaska native",
    "middle eastern", "biracial", "multiracial",
}
# head nouns indicating that a preceding eponym is used medically (Bell's PALSY, Crohn's DISEASE ...)
MEDICAL_HEADS = {
    "disease", "diseases", "syndrome", "palsy", "sign", "signs", "test", "maneuver", "manoeuvre",
    "fracture", "phenomenon", "disorder", "lymphoma", "sarcoma", "tumor", "tumour", "cyst", "nodes",
    "node", "esophagus", "thyroiditis", "contracture", "neuroma", "deformity", "score", "scale",
    "catheter", "monitor", "chorea", "dementia", "body", "bodies", "triad", "injury", "lesion",
    "lesions", "tube", "pupil", "ulcer", "hernia", "diverticulum", "criteria", "procedure", "operation",
    "reflex", "virus", "fever", "encephalitis", "tendon", "tendinitis", "tendonitis", "tendinopathy",
    "rupture", "angina", "block", "colitis", "flare", "arthritis", "cardiomyopathy", "tenosynovitis",
    "parkinsonism", "plus", "crisis", "ophthalmopathy", "orbitopathy", "kyphosis", "spots", "joint",
    "arthropathy", "sounds", "coma", "position", "gait", "type", "medication", "medications", "meds", "testing",
}

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_U = "|".join(_UNITS[1:])
_BELOW100 = rf"(?:(?:{'|'.join(_TENS)})(?:[\s-]+(?:{_U}))?|{'|'.join(_TEENS)}|{_U}|zero)"
NUM_WORDS_RE = rf"(?:(?:(?:a|one)\s+)?hundred(?:\s+and)?(?:\s+{_BELOW100})?|{_BELOW100})"
DECIMAL_WORDS_RE = rf"{NUM_WORDS_RE}(?:\s+point(?:\s+(?:{_U}|zero|oh))+)?"
NUMBER_RE = rf"(?:\d+(?:\.\d+)?|{DECIMAL_WORDS_RE})"
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}
ORDINAL_WORDS_RE = rf"(?:(?:twenty|thirty)[\s-]+)?(?:{'|'.join(ORDINAL_WORDS)})"


def words_to_number(text: str) -> float | None:
    """Parse '62', '8.3', 'sixty two', 'sixty-two', 'eight point three', 'one hundred and two'."""
    s = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    toks = [t for t in re.split(r"[\s-]+", s) if t and t != "and"]
    if not toks:
        return None
    integer, decimals, in_decimal, seen = 0, "", False, False
    for t in toks:
        if t == "point":
            in_decimal = True
            continue
        if in_decimal:
            if t in ("oh", "zero"):
                decimals += "0"
            elif t in _UNITS:
                decimals += str(_UNITS.index(t))
            else:
                return None
            continue
        if t in ("a",):
            continue
        if t == "hundred":
            integer = max(integer, 1) * 100
        elif t in _TENS:
            integer += _TENS[t]
        elif t in _TEENS:
            integer += 10 + _TEENS.index(t)
        elif t in _UNITS:
            integer += _UNITS.index(t)
        else:
            return None
        seen = True
    if not seen:
        return None
    return float(f"{integer}.{decimals}") if decimals else float(integer)


def number_to_words(value: float, decimals: int) -> str:
    """Render a number the way ASR transcripts spell it ('eight point one', 'sixty two')."""
    def below_1000(n: int) -> str:
        parts = []
        if n >= 100:
            parts.append(f"{_UNITS[n // 100]} hundred")
            n %= 100
            if n:
                parts.append("and")
        if n >= 20:
            tens = [k for k, v in _TENS.items() if v == n // 10 * 10 and k != "fourty"][0]
            parts.append(tens + (f" {_UNITS[n % 10]}" if n % 10 else ""))
        elif n >= 10:
            parts.append(_TEENS[n - 10])
        elif n > 0 or not parts:
            parts.append(_UNITS[n])
        return " ".join(parts)

    integer = int(abs(value))
    text = below_1000(integer) if integer < 1000 else str(integer)
    if decimals > 0:
        frac = f"{abs(value):.{decimals}f}".split(".")[1]
        text += " point " + " ".join(_UNITS[int(d)] for d in frac)
    return text


def _lines(name: str):
    with open(RES_DIR / name, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                yield line


@dataclass
class Lexicon:
    first_names: dict[str, float]   # name -> peak popularity (% of people/births)
    surnames: dict[str, int]        # name -> US census 2010 count
    common: dict[str, float]        # word -> zipf frequency
    drugs: set[str]
    icd_vocab: dict[str, int]
    clinical: set[str]
    eponyms: dict[str, set[str]]    # eponym -> typical following head nouns
    occupations: dict[str, str]     # occupation -> coarse category
    places: dict[str, str]          # place -> type

    def zipf(self, w: str) -> float:
        return self.common.get(w.lower(), 0.0)

    def name_ratio(self, w: str) -> float:
        """Estimated share of a word's occurrences that are personal-name usages.

        p_name ~ (share of tokens that are person names) * (popularity of this name);
        p_word ~ 10**(zipf-9) from general English frequencies. Ratio > ~0.5 means the
        token is predominantly a name (andrew, perez), < 0.1 predominantly a word (will, hope).
        """
        w = w.lower().removesuffix("'s")
        if w in NOT_NAMES:
            return 0.0
        p_first = 0.005 * self.first_names.get(w, 0.0) / 100.0
        p_sur = 0.003 * self.surnames.get(w, 0) / 3.1e8
        p_name = max(p_first, p_sur)
        if p_name == 0.0:
            return 0.0
        z = self.zipf(w)
        p_word = 10 ** (z - 9) if z > 0 else 1e-9
        return min(1.0, p_name / p_word)

    def in_name_lexicon(self, w: str) -> bool:
        w = w.lower()
        return w in self.first_names or w in self.surnames

    def is_drug(self, w: str) -> bool:
        return w.lower() in self.drugs

    def is_clinical(self, w: str) -> bool:
        return w.lower() in self.clinical

    def medical_strength(self, w: str) -> int:
        """2 = strong medical evidence (drug / clinical term / rare ICD-10-CM word / clinical
        morphology), 1 = weak (common ICD vocabulary), 0 = none. Eponyms are handled separately."""
        w = w.lower().removesuffix("'s")
        hit = _MED_CACHE.get(w)
        if hit is not None:
            return hit
        z = self.zipf(w)
        if w in self.drugs or w in self.clinical:
            hit = 2
        elif w in self.icd_vocab and w not in self.eponyms and z < 3.0:
            hit = 2  # "melena", "dysphagia": rare in English, present in ICD-10-CM descriptions
        elif z < 3.0 and len(w) > 6 and _CLINICAL_MORPHOLOGY.search(w) and self.name_ratio(w) < 0.3:
            hit = 2  # "mitotic", "cholecystectomy", "arthroplasty", "bradykinesia", "canalith" ...
        elif z < 3.5 and len(w) > 6 and self.name_ratio(w) < 0.3 and any(
                w.endswith(suf) and w[:-len(suf)] in self.eponyms for suf in ("ian", "ism", "oid", "ic")):
            hit = 2  # adjective / noun derived from an eponym: "parkinsonian", "cushingoid", "addisonian"
        elif z < 3.0 and len(w) > 7 and self.name_ratio(w) < 0.3 and (
                any(w.startswith(p) and len(w) - len(p) >= 6 and self.medical_strength(w[len(p):]) >= 1 for p in _CLINICAL_PREFIXES)
                or any(w.endswith(suf) and self.medical_strength(stem) == 2
                       for suf in ("al", "ic", "ous", "ar") for stem in (w[:-len(suf)], w[:-len(suf)] + "e", w[:-len(suf)] + "a")
                       if len(stem) > 5)):
            hit = 2  # derived clinical word: "supratherapeutic", "lactational", "dermatomal"
        elif "-" in w and all(len(p) > 3 and self.medical_strength(p) == 2 for p in w.split("-")):
            hit = 2  # combination product / compound term: "budesonide-formoterol", "amlodipine-benazepril"
        elif self.icd_vocab.get(w, 0) >= 3 and self.name_ratio(w) < 0.5:
            hit = 1
        elif len(w) > 4 and w.endswith("s") and self.name_ratio(w) < 0.3:
            # plural of a clinical term ("myalgias", "cataracts", "carotids")
            sing = [w[:-3] + "y"] if w.endswith("ies") else []
            sing += [w[:-1]] + ([w[:-2]] if w.endswith("es") else [])
            # professions in the plural name practices ("Optimum Optometrists"): no medical evidence
            hit = 0 if w[:-1].endswith(("ist", "ian")) else max(self.medical_strength(s) for s in sing)
        else:
            hit = 0
        _MED_CACHE[w] = hit
        return hit


_MED_CACHE: dict[str, int] = {}
_CLINICAL_PREFIXES = ("supra", "sub", "infra", "hyper", "hypo", "peri", "intra", "extra", "retro", "trans", "anti",
                      "post", "pre", "para", "poly", "micro", "macro", "neo", "inter", "non")
_CLINICAL_MORPHOLOGY = re.compile(
    r"(?:itis|osis|oses|emia|aemia|algia|ectomy|ostomy|otomy|plasty|scopy|scopic|ology|ologist|opathy|"
    r"oma|omas|uria|penia|cytosis|cytes?|genic|otic|trophy|trophic|plegia|paresis|rrhea|rrhoea|rrhage|lysis|"
    r"kinesia|kinesis|phasia|phagia|pnea|pnoea|esthesia|aesthesia|cardia|liths?|iasis|ectasia|ectasis|malacia|"
    r"megaly|spasm|taxia|opia|plasia|"
    r"pril|sartan|statin|olol|azole|mycin|cillin|cycline|floxacin|dipine|tidine|prazole|parin|gliptin|"
    r"gliflozin|glutide|mab|tinib|vir|thiazide|semide|codone|morphine|caine)$")


@lru_cache(maxsize=1)
def load_lexicon() -> Lexicon:
    first = {}
    for line in _lines("first_names.tsv"):
        n, p = line.split("\t")
        first[n] = float(p)
    surnames = {}
    for line in _lines("surnames.tsv"):
        n, c = line.split("\t")
        surnames[n] = int(c)
    common = {}
    for line in _lines("common_words.tsv"):
        w, z = line.split("\t")
        common[w] = float(z)
    drugs = set(_lines("drugs.txt"))
    icd_vocab = {}
    for line in _lines("icd_vocab.tsv"):
        w, c = line.split("\t")
        icd_vocab[w] = int(c)
    clinical = {t.strip().lower() for t in _lines("clinical_terms.txt")}
    eponyms: dict[str, set[str]] = {}
    for line in _lines("eponyms.tsv"):
        ep, _, heads = (line.split("\t") + ["", ""])[:3]
        eponyms.setdefault(ep, set()).update(h for h in heads.split(",") if h)
    for line in _lines("eponyms_extra.tsv"):
        ep, heads = line.split("\t")
        eponyms.setdefault(ep, set()).update(h for h in heads.split(",") if h)
    occupations = dict(line.split("\t") for line in _lines("occupations.tsv"))
    places = dict(line.split("\t") for line in _lines("places.tsv"))
    # drug tokens that are predominantly personal names are not reliable medical evidence
    drugs = {d for d in drugs if not (d in first and first[d] >= 0.05 and d not in clinical)}
    return Lexicon(first, surnames, common, drugs, icd_vocab, clinical, eponyms, occupations, places)


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]*[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]|[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]|\d+(?:[.,/:-]\d+)*|[^\sA-Za-zÀ-ÖØ-öø-ɏḀ-ỿ\d]")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Whitespace/punctuation tokenizer returning (token, start, end) with character offsets."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def is_cased(text: str) -> bool:
    letters = sum(c.isalpha() for c in text[:5000])
    upper = sum(c.isupper() for c in text[:5000])
    return letters > 0 and upper / letters > 0.01


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
