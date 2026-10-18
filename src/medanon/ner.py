"""Statistical NER detectors: spaCy (via Presidio) and optional transformer models.

* ``SpacyPresidioDetector`` - spaCy ``en_core_web_lg`` entities + Presidio's validated pattern
  recognizers (credit cards, IBAN, IPs, phone, e-mail ...). For lowercase ASR dialogue we run NER
  on a *truecased view* of the text (same length, so offsets map 1:1): tokens that the physician's
  cased note writes as proper nouns, name-dominant tokens and gazetteer places are capitalised.
* ``MLDetector`` - pluggable transformer NER (GLiNER, GLiNER2, HF token classification), used for
  the comparison study and as an optional "accurate" profile. All run locally on CPU or <=4 GB GPU.
"""
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("TLDEXTRACT_PUBLIC_SUFFIX_LIST_URLS", "")  # Presidio e-mail/URL: offline PSL snapshot

from .lexicon import MONTHS, WEEKDAYS, Lexicon, is_cased
from .spans import Span

# Presidio's spaCy engine already renames labels (GPE/LOC/FAC -> LOCATION, ORG -> ORGANIZATION,
# NORP -> NRP); raw spaCy names are kept too in case the mapping configuration changes.
SPACY_MAP = {"PERSON": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION", "LOCATION": "LOCATION",
             "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION", "NORP": "NATIONALITY", "NRP": "NATIONALITY"}
PRESIDIO_PATTERNS = {"EMAIL_ADDRESS": "CONTACT", "PHONE_NUMBER": "CONTACT", "URL": "CONTACT",
                     "IP_ADDRESS": "CONTACT", "US_SSN": "ID", "CREDIT_CARD": "ID", "IBAN_CODE": "ID",
                     "MEDICAL_LICENSE": "ID", "US_PASSPORT": "ID", "UK_NHS": "ID"}
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]*")


def note_case_map(note: str, lex: Lexicon) -> dict[str, str]:
    """Proper nouns as written in the cased note (not sentence-initial, not all-caps headers)."""
    out = {}
    for m in re.finditer(r"(?<=[a-z,;:]\s)([A-ZÀ-ÖØ-ÞĀ-ɏḀ-ỿ][a-zß-öø-ÿĀ-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]*)", note):
        w = m.group(1)
        if lex.zipf(w.lower()) < 4.5:
            out[w.lower()] = w
    return out


def truecase(text: str, lex: Lexicon, case_map: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if lw in case_map and len(case_map[lw]) == len(w):
            return case_map[lw]
        if lw == "i" or lw in WEEKDAYS or (lw in MONTHS and len(lw) > 3 and lw != "may"):
            return w.capitalize()
        if lex.name_ratio(lw) >= 0.5 and lex.medical_strength(lw) < 2:
            return w.capitalize()
        if lw in lex.places and lex.zipf(lw) < 4.3:
            return w.capitalize()
        return w

    out = _WORD.sub(repl, text)
    return re.sub(r"(^|[.?!]\s+|\]\s+)([a-zß-öø-ÿĀ-ɏḀ-ỿ])", lambda m: m.group(1) + m.group(2).upper(), out, flags=re.M)


class SpacyPresidioDetector:
    def __init__(self, lex: Lexicon, model: str = "en_core_web_lg", use_presidio_patterns: bool = True,
                 truecase_dialogue: bool = True, min_score: float = 0.5):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        conf = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}],
                "ner_model_configuration": {"labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT",
                                                                 "DATE", "TIME", "LANGUAGE", "LAW", "WORK_OF_ART"]}}
        self.engine = NlpEngineProvider(nlp_configuration=conf).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=self.engine, supported_languages=["en"]) if use_presidio_patterns else None
        self.lex = lex
        self.truecase_dialogue = truecase_dialogue
        self.min_score = min_score

    def detect(self, text: str, field: str, case_map: dict[str, str] | None = None) -> list[Span]:
        cased = is_cased(text)
        view = text if cased or not self.truecase_dialogue else truecase(text, self.lex, case_map or {})
        artifacts = self.engine.process_text(view, "en")
        out = []
        for ent in artifacts.entities:
            label = SPACY_MAP.get(ent.label_)
            if label is None:
                continue
            s, e = ent.start_char, ent.end_char
            if text[s:e].lower().startswith("the "):
                s += 4
            out.append(Span(field, s, e, label, f"spacy:{ent.label_}", 0.75 if cased else 0.6, text[s:e],
                            meta={"pos": [t.pos_ for t in ent if t.idx >= s]},  # the tagger's view of the same tokens
                            reasons=[f"spaCy NER labelled it {ent.label_}" + ("" if cased else " (on truecased dialogue)")]))
        if self.analyzer is not None:
            for r in self.analyzer.analyze(text=view, language="en", entities=list(PRESIDIO_PATTERNS),
                                           nlp_artifacts=artifacts, score_threshold=self.min_score):
                out.append(Span(field, r.start, r.end, PRESIDIO_PATTERNS[r.entity_type], f"presidio:{r.entity_type}",
                                r.score, text[r.start:r.end], meta={"kind": r.entity_type.lower()},
                                reasons=[f"Presidio {r.entity_type} recognizer (score {r.score:.2f})"]))
        return out


# ---------------------------------------------------------------------------------------- ML NER
ML_LABELS = {  # natural-language label for zero-shot models -> canonical label
    "person": "PERSON", "location": "LOCATION", "organization": "ORGANIZATION",
    "occupation": "OCCUPATION", "address": "ADDRESS", "phone number": "CONTACT", "email": "CONTACT",
    "medical record number": "ID",
}
HF_MAP = {"PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION", "GPE": "LOCATION", "ORG": "ORGANIZATION",
          "PATIENT": "PERSON", "STAFF": "PERSON", "HCW": "PERSON", "HOSP": "ORGANIZATION", "PATORG": "ORGANIZATION",
          "VENDOR": "ORGANIZATION", "ID": "ID", "PHONE": "CONTACT", "EMAIL": "CONTACT"}
# label schemas of the clinical PII models evaluated in MODEL_EXPERIMENTS.md (ages and generic dates are left
# to the rules, which band/replace them with clinical safeguards; gender/race are not removed)
LABEL_MAPS = {
    "healthcare-brain": {"PATIENT_NAME": "PERSON", "DOCTOR_NAME": "PERSON", "HOSPITAL": "ORGANIZATION",
                         "CITY": "LOCATION", "STATE": "LOCATION", "COUNTRY": "LOCATION", "ZIP": "ADDRESS",
                         "ADDRESS": "ADDRESS", "MRN": "ID", "ID_NUMBER": "ID", "SSN": "ID", "USERNAME": "ID",
                         "PHONE": "CONTACT", "FAX": "CONTACT", "EMAIL": "CONTACT", "URL": "CONTACT", "IP_ADDRESS": "CONTACT"},
    "openmed-pii": {"FIRST_NAME": "PERSON", "LAST_NAME": "PERSON", "USER_NAME": "ID", "COMPANY_NAME": "ORGANIZATION",
                    "CITY": "LOCATION", "STATE": "LOCATION", "COUNTY": "LOCATION", "COUNTRY": "LOCATION",
                    "COORDINATE": "LOCATION", "STREET_ADDRESS": "ADDRESS", "POSTCODE": "ADDRESS",
                    "MEDICAL_RECORD_NUMBER": "ID", "HEALTH_PLAN_BENEFICIARY_NUMBER": "ID", "ACCOUNT_NUMBER": "ID",
                    "SSN": "ID", "CUSTOMER_ID": "ID", "EMPLOYEE_ID": "ID", "UNIQUE_ID": "ID", "TAX_ID": "ID",
                    "CERTIFICATE_LICENSE_NUMBER": "ID", "LICENSE_PLATE": "ID", "VEHICLE_IDENTIFIER": "ID",
                    "DEVICE_IDENTIFIER": "ID", "CREDIT_DEBIT_CARD": "ID", "PHONE_NUMBER": "CONTACT",
                    "FAX_NUMBER": "CONTACT", "EMAIL": "CONTACT", "URL": "CONTACT", "IPV4": "CONTACT", "IPV6": "CONTACT",
                    "OCCUPATION": "OCCUPATION"},
}


def _chunks(text: str, max_chars: int):
    """Split on line boundaries into chunks <= max_chars (long lines are split on sentence ends)."""
    buf_start, pieces = 0, []
    for line in re.finditer(r"[^\n]*\n?", text):
        if not line.group(0):
            continue
        s, e = line.start(), line.end()
        if e - buf_start > max_chars and s > buf_start:
            pieces.append((buf_start, s))
            buf_start = s
        while e - buf_start > max_chars:
            cut = text.rfind(". ", buf_start, buf_start + max_chars)
            cut = cut + 2 if cut > buf_start else buf_start + max_chars
            pieces.append((buf_start, cut))
            buf_start = cut
    if buf_start < len(text):
        pieces.append((buf_start, len(text)))
    return pieces


def resolve_model(model: str, allow_download: bool) -> tuple[str, bool]:
    """Return (local directory, downloaded_now). Local paths are used as is; Hugging Face ids are
    looked up in the local cache first and fetched only if allowed (one-time setup)."""
    if os.path.isdir(model):
        return model, False
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, local_files_only=True), False
    except Exception:  # noqa: BLE001 - not cached yet
        if not allow_download:
            raise RuntimeError(f"model '{model}' is not in the local Hugging Face cache and downloads are disabled "
                               f"(--offline). Run `medanon prepare-models` once with internet access.") from None
    print(f"[medanon] one-time download of model weights '{model}' into the local cache ...", file=sys.stderr)
    return snapshot_download(model), True


def _windows(text: str, start: int, end: int, max_chars: int, overlap: int = 150) -> list[tuple[int, int]]:
    """Split [start, end) into windows of <= max_chars that end at whitespace and overlap by
    `overlap` characters, so an entity on a window boundary is seen whole at least once."""
    if end - start <= max_chars:
        return [(start, end)]
    out, s = [], start
    while s < end:
        e = min(end, s + max_chars)
        if e < end:
            cut = text.rfind(" ", s + max_chars // 2, e)
            e = cut if cut > s else e
        out.append((s, e))
        if e >= end:
            break
        nxt = text.find(" ", max(s + 1, e - overlap), e)
        s = nxt + 1 if nxt != -1 else e
    return out


def _pack(regions: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Merge sorted (start, end) regions into windows of at most max_chars (keeps model calls few).
    Merged windows may include the text between regions, which only adds context."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(regions):
        if out and e - out[-1][0] <= max_chars:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


class MLDetector:
    """kind in {'gliner', 'gliner2', 'hf'}; model is a Hugging Face id or local path."""

    def __init__(self, kind: str, model: str, threshold: float = 0.5, max_chars: int = 1000,
                 labels: dict[str, str] | None = None, device: str = "cpu", allow_download: bool = False,
                 label_map: str | None = None):
        self.kind, self.model_name, self.threshold, self.max_chars = kind, model, threshold, max_chars
        self.labels = labels or ML_LABELS
        self.label_map = label_map
        self.device = device
        self.local_path, self.downloaded = resolve_model(model, allow_download)
        self.revision = os.path.basename(self.local_path.rstrip("/\\"))
        self._load(kind, self.local_path, device)  # always from a local directory: no network here

    def _load(self, kind: str, path: str, device: str) -> None:
        if kind == "gliner":
            from gliner import GLiNER
            self.model = GLiNER.from_pretrained(path, local_files_only=True)
            self.model.to(device)
        elif kind == "gliner2":
            from gliner2 import GLiNER2  # local runtime only; the gliner2 API client is never used
            self.model = GLiNER2.from_pretrained(path)
        elif kind == "hf":
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            from transformers import logging as hf_logging
            from transformers import pipeline

            hf_logging.set_verbosity_error()
            self.model = pipeline("token-classification", model=path, aggregation_strategy="first",
                                  device=-1 if device == "cpu" else device)
        else:
            raise ValueError(f"unknown ML detector kind {kind}")

    def weights_sha256(self) -> str:
        import hashlib

        files = sorted(f for f in os.listdir(self.local_path) if f.endswith((".safetensors", ".bin", ".pt", ".onnx")))
        h = hashlib.sha256()
        for f in files:
            with open(os.path.join(self.local_path, f), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
        return h.hexdigest() if files else "n/a"

    def _predict(self, chunk: str) -> list[tuple[int, int, str, float]]:
        if self.kind == "gliner":
            preds = self.model.predict_entities(chunk, list(self.labels), threshold=self.threshold)
            return [(p["start"], p["end"], self.labels[p["label"]], float(p["score"])) for p in preds]
        if self.kind == "gliner2":
            res = self.model.extract_entities(chunk, list(self.labels), threshold=self.threshold,
                                              include_confidence=True, include_spans=True)
            out = []
            for lab, items in (res.get("entities") or {}).items():
                for it in items:
                    if isinstance(it, dict) and "start" in it:
                        out.append((it["start"], it["end"], self.labels.get(lab, lab), float(it.get("confidence", 1.0))))
            return out
        out = []
        mapping = LABEL_MAPS.get(self.label_map, HF_MAP)
        for p in self.model(chunk):
            lab = mapping.get(p["entity_group"].upper())
            if lab and p["score"] >= self.threshold:
                out.append((p["start"], p["end"], lab, float(p["score"])))
        return out

    def detect(self, text: str, field: str, regions: list[tuple[int, int]] | None = None) -> list[Span]:
        """Run on the whole text, or only on `regions` (cascade mode: uncertain dialogue turns)."""
        out, seen = [], set()
        pieces = _chunks(text, self.max_chars) if regions is None else _pack(regions, self.max_chars)
        for p0, p1 in pieces:
            for s0, e0 in _windows(text, p0, p1, self.max_chars):  # long turns: overlapping windows
                for s, e, lab, score in self._predict(text[s0:e0]):
                    key = (s0 + s, s0 + e, lab)
                    if lab not in {"PERSON", "LOCATION", "ORGANIZATION", "OCCUPATION", "ADDRESS", "CONTACT", "ID"} or key in seen:
                        continue
                    seen.add(key)
                    out.append(Span(field, s0 + s, s0 + e, lab, f"ml:{self.kind}", score, text[s0 + s:s0 + e],
                                    reasons=[f"{self.kind} model {self.model_name} predicted {lab} ({score:.2f})"]))
        return out
