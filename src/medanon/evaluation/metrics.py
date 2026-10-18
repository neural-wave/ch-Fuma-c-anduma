"""Local evaluation on real data without gold labels (privacy / utility / integrity / efficiency).

None of these numbers is an official Bosch score. They are proxies chosen to resemble the
announced metrics: "proportion of removed privacy-related information" (here: what an
*independent* off-the-shelf detector finds in the original and whether it survives), and
"ICD-10 prediction F1 + medical concept recall" (here: a TF-IDF coder trained on the dev split and
a lexicon concept extractor applied identically to original and anonymised text).
All outputs are aggregate numbers; no text is printed or stored.
"""
from __future__ import annotations

import re
from collections import Counter

from ..lexicon import load_lexicon

TEXT_FIELDS = ("dialogue", "note")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]\d+[A-Za-z\d]*")
_STOP = {"the", "and", "with", "without", "for", "of", "unspecified", "other", "due", "to", "in", "or", "not",
         "elsewhere", "classified", "type", "site", "initial", "encounter", "subsequent", "side", "left", "right"}


# ------------------------------------------------------------------------------------- integrity
def integrity(originals: list[dict], anonymized: list[dict]) -> dict:
    by_id = {r.get("encounter_id"): r for r in anonymized}
    issues = Counter()
    for o in originals:
        a = by_id.get(o.get("encounter_id"))
        if a is None:
            issues["missing_record"] += 1
            continue
        if set(a) != set(o):
            issues["key_mismatch"] += 1
        if a.get("icd10_codes") != o.get("icd10_codes"):
            issues["icd10_changed"] += 1
        for k in o:
            if k not in TEXT_FIELDS and a.get(k) != o.get(k):
                issues[f"non_text_field_changed:{k}"] += 1
            if k in TEXT_FIELDS and not isinstance(a.get(k), str):
                issues[f"text_field_not_string:{k}"] += 1
    return {"records": len(originals), "issues": dict(issues), "ok": not issues}


def validate_deliverable(original_path: str, anonymized_path: str) -> dict:
    """Line-level validation of the submission file (stricter than `integrity`): every line parses,
    same record count and order, unique ids, identical key order, non-text fields byte-identical,
    text fields are strings, speaker tags and line breaks preserved, UTF-8."""
    import json

    problems: list[str] = []
    with open(original_path, encoding="utf-8") as f:
        orig = [json.loads(line) for line in f if line.strip()]
    anon = []
    with open(anonymized_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                problems.append(f"line {i}: empty line")
                continue
            try:
                anon.append(json.loads(line))
            except json.JSONDecodeError as exc:
                problems.append(f"line {i}: invalid JSON ({exc.msg})")
    ids = [r.get("encounter_id") for r in anon]
    if len(anon) != len(orig):
        problems.append(f"record count {len(anon)} != {len(orig)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate encounter_id")
    for i, (o, a) in enumerate(zip(orig, anon)):
        if o.get("encounter_id") != a.get("encounter_id"):
            problems.append(f"record {i}: order/id mismatch")
            continue
        if list(o) != list(a):
            problems.append(f"record {i}: keys differ")
        for k in o:
            if k in TEXT_FIELDS:
                if not isinstance(a.get(k), str):
                    problems.append(f"record {i}: {k} not a string")
                    continue
                if o[k].count("\n") != a[k].count("\n"):
                    problems.append(f"record {i}: {k} line structure changed")
                for tag in set(re.findall(r"\[[a-z_]+\]", o[k])):
                    if o[k].count(tag) != a[k].count(tag):
                        problems.append(f"record {i}: speaker tag {tag} count changed")
            elif a.get(k) != o.get(k):
                problems.append(f"record {i}: non-text field '{k}' changed")
    return {"ok": not problems, "records": len(anon), "expected": len(orig), "problems": problems[:50]}


# ------------------------------------------------------------------------------------- utility
def medical_concepts(text: str) -> Counter:
    """Lexicon concept extractor: drug names, curated clinical terms, eponyms and ICD-10-CM
    vocabulary words that are not ordinary English (zipf < 4) - plus their bigrams with ICD words."""
    lex = load_lexicon()
    toks = [t.lower() for t in _WORD.findall(text)]
    out = Counter()
    for i, t in enumerate(toks):
        t2 = t.removesuffix("'s")
        med = lex.medical_strength(t2) == 2 or t2 in lex.eponyms or (t2 in lex.icd_vocab and lex.zipf(t2) < 4.0 and t2 not in _STOP)
        if med:
            out[t2] += 1
            if i + 1 < len(toks) and toks[i + 1] in lex.icd_vocab and toks[i + 1] not in _STOP:
                out[f"{t2} {toks[i + 1]}"] += 1
    return out


def concept_recall(originals: list[dict], anonymized: list[dict]) -> dict:
    by_id = {r.get("encounter_id"): r for r in anonymized}
    per_rec, mention_tot, mention_kept = [], 0, 0
    lost = Counter()
    for o in originals:
        a = by_id[o.get("encounter_id")]
        co = sum((medical_concepts(o[f]) for f in TEXT_FIELDS), Counter())
        ca = sum((medical_concepts(a[f]) for f in TEXT_FIELDS), Counter())
        if not co:
            continue
        per_rec.append(len(set(co) & set(ca)) / len(co))
        mention_tot += sum(co.values())
        mention_kept += sum(min(v, ca[k]) for k, v in co.items())
        for k in set(co) - set(ca):
            lost[k] += 1
    return {"concept_set_recall": round(sum(per_rec) / max(1, len(per_rec)), 4),
            "concept_mention_recall": round(mention_kept / max(1, mention_tot), 4),
            "lost_concepts_top": lost.most_common(15)}


def icd_term_retention(originals: list[dict], anonymized: list[dict]) -> dict:
    """Of the ICD-10 description words that occur in the original text, how many still occur."""
    by_id = {r.get("encounter_id"): r for r in anonymized}
    tot = kept = 0
    for o in originals:
        a = by_id[o.get("encounter_id")]
        otext = " ".join(o[f] for f in TEXT_FIELDS).lower()
        atext = " ".join(a[f] for f in TEXT_FIELDS).lower()
        words = {w for item in o.get("icd10_codes", []) for w in _WORD.findall(item["description"].lower())
                 if len(w) > 3 and w not in _STOP}
        for w in words:
            if re.search(rf"\b{re.escape(w)}", otext):
                tot += 1
                kept += bool(re.search(rf"\b{re.escape(w)}", atext))
    return {"icd_terms_in_text": tot, "retained": round(kept / max(1, tot), 4)}


def _doc(r: dict) -> str:
    return (r.get("dialogue", "") + "\n" + r.get("note", "")).lower()


def icd_proxy_f1(train: list[dict], test_orig: list[dict], test_anon: list[dict], train_anon: list[dict] | None = None) -> dict:
    """TF-IDF + one-vs-rest logistic regression ICD-10 coder (codes seen >= 2x in training).
    Reports micro-F1 on original vs anonymised test text; the *difference* is the utility signal."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    def codes(r):
        return [c["code"] for c in r.get("icd10_codes", [])]

    freq = Counter(c for r in train for c in codes(r))
    label_space = sorted(c for c, n in freq.items() if n >= 2)
    mlb = MultiLabelBinarizer(classes=label_space)
    y_train = mlb.fit_transform([[c for c in codes(r) if c in label_space] for r in train])
    y_test = mlb.transform([[c for c in codes(r) if c in label_space] for r in test_orig])

    def fit_predict(train_docs, test_docs):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z0-9]+\b")
        x = vec.fit_transform(train_docs)
        clf = OneVsRestClassifier(LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced"))
        clf.fit(x, y_train)
        prob = clf.predict_proba(vec.transform(test_docs))
        pred = (prob >= 0.35).astype(int)
        for i in range(len(pred)):  # every encounter has at least one diagnosis
            if not pred[i].any():
                pred[i, prob[i].argmax()] = 1
        return f1_score(y_test, pred, average="micro", zero_division=0)

    orig_by = {r["encounter_id"]: r for r in test_anon}
    anon_aligned = [orig_by[r["encounter_id"]] for r in test_orig]
    res = {"label_space": len(label_space),
           "f1_original": round(fit_predict([_doc(r) for r in train], [_doc(r) for r in test_orig]), 4),
           "f1_anonymized": round(fit_predict([_doc(r) for r in train], [_doc(r) for r in anon_aligned]), 4)}
    if train_anon is not None:
        res["f1_trained_and_tested_on_anonymized"] = round(fit_predict([_doc(r) for r in train_anon], [_doc(r) for r in anon_aligned]), 4)
    res["delta"] = round(res["f1_anonymized"] - res["f1_original"], 4)
    return res


# ------------------------------------------------------------------------------------- privacy
class ReferenceDetector:
    """Independent PII detector = *vanilla* Presidio (default recognisers, spaCy en_core_web_lg).
    It knows nothing about our rules, lexicons or propagation."""

    ENTITIES = ["PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "NRP", "URL"]

    def __init__(self):
        import os

        os.environ.setdefault("TLDEXTRACT_PUBLIC_SUFFIX_LIST_URLS", "")
        from presidio_analyzer import AnalyzerEngine

        self.engine = AnalyzerEngine()

    def find(self, text: str) -> list[tuple[str, str]]:
        res = self.engine.analyze(text=text, language="en", entities=self.ENTITIES, score_threshold=0.5)
        return [(r.entity_type, text[r.start:r.end]) for r in res]


_RELATIVE_TIME = re.compile(
    r"^(?:today|yesterday|tomorrow|tonight|now|daily|weekly|monthly|yearly|annually|nightly|"
    r"(?:the\s+)?(?:last|past|next|this|few|several|couple|first|second)\b.*|.*\b(?:ago|days?|weeks?|months?|years?|"
    r"hours?|minutes?|times?|morning|evening|night|afternoon|weekend|decades?|lately|recently|old)$)", re.I)
# A temporal expression is privacy-relevant when it anchors to the calendar (HIPAA: all date
# elements except the year, plus ages); durations, frequencies and seasons are not.
_CALENDAR_ANCHOR = re.compile(
    r"\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}|\b\d{1,2}/\d{4}\b|\b(?:19|20)\d{2}\b|year[- ]old|years?\s+old|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"christmas|thanksgiving|easter|halloween|new year|fourth of july|\b\d{1,2}(?:st|nd|rd|th)\b", re.I)


def leakage_audit(originals: list[dict], anonymized: list[dict], ref: ReferenceDetector | None = None) -> dict:
    """Proportion of reference-detected private items (in the original) absent from the output.

    Items are matched case-insensitively at word boundaries. DATE_TIME items that are durations or
    relative expressions ("three weeks", "yesterday") are reported separately because they carry no
    identifying calendar information and are intentionally kept for clinical utility; items that are
    medical vocabulary (drug names tagged PERSON, eponyms) are likewise reported separately.
    """
    lex = load_lexicon()
    ref = ref or ReferenceDetector()
    by_id = {r.get("encounter_id"): r for r in anonymized}
    tot, removed = Counter(), Counter()
    excluded = Counter()
    for o in originals:
        a = by_id[o.get("encounter_id")]
        for f in TEXT_FIELDS:
            seen = set()
            for etype, surf in ref.find(o[f]):
                key = (etype, surf.lower().strip())
                if key in seen or len(key[1]) < 2:
                    continue
                seen.add(key)
                words = [w.lower() for w in _WORD.findall(surf)]
                if etype == "DATE_TIME" and not _CALENDAR_ANCHOR.search(surf):
                    excluded["DATE_TIME:no calendar anchor (duration/frequency/season/relative)"] += 1
                    continue
                if words and all(lex.medical_strength(w.removesuffix("'s")) == 2 or w.removesuffix("'s") in lex.eponyms
                                 for w in words):
                    excluded[f"{etype}:medical-term"] += 1
                    continue
                gone = not re.search(rf"(?<![A-Za-z0-9]){re.escape(surf.strip())}(?![A-Za-z0-9])", a[f], re.I)
                tot[etype] += 1
                removed[etype] += gone
                tot["ALL"] += 1
                removed["ALL"] += gone
    return {"proportion_removed": {k: round(removed[k] / tot[k], 4) for k in sorted(tot)},
            "items": dict(tot), "excluded_from_score": dict(excluded)}


def name_seed_leakage(originals: list[dict], anonymized: list[dict]) -> dict:
    """High-precision seeds: names after Mr./Mrs./Ms./Dr. and subjects of 'X is a N-year-old' in the
    cased note. Counts records where any seed token survives anywhere in the output, ignoring
    occurrences that are eponym + medical head (e.g. surname Parkinson vs. Parkinson's disease)."""
    from ..lexicon import MEDICAL_HEADS

    lex = load_lexicon()
    by_id = {r.get("encounter_id"): r for r in anonymized}
    seeds_total = seeds_leaked = recs_with_leak = 0
    for o in originals:
        a = by_id[o.get("encounter_id")]
        note = o.get("note", "")
        seeds = set(re.findall(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-z'\-]+)", note))
        for m in re.finditer(r"(?:^|\n)((?:[A-Z][a-z'\-]+\s){0,2}[A-Z][a-z'\-]+) is an? \d+-year-old", note):
            seeds.update(w for w in m.group(1).split() if w.lower() not in {"the", "patient", "this"})
        out_text = a.get("dialogue", "") + "\n" + a.get("note", "")
        leaked_here = False
        for s in seeds:
            seeds_total += 1
            for m in re.finditer(rf"(?<![A-Za-z]){re.escape(s)}(?![A-Za-z])", out_text, re.I):
                nxt = re.findall(r"[a-z]+", out_text[m.end():m.end() + 20].lower())[:2]
                if s.lower() in lex.eponyms and any(n in MEDICAL_HEADS or n in lex.eponyms.get(s.lower(), ()) or n == "s" for n in nxt):
                    continue
                seeds_leaked += 1
                leaked_here = True
                break
        recs_with_leak += leaked_here
    return {"seed_names": seeds_total, "seed_names_leaked": seeds_leaked,
            "records_with_leak": recs_with_leak, "records": len(originals)}


def edit_stats(originals: list[dict], anonymized: list[dict]) -> dict:
    by_id = {r.get("encounter_id"): r for r in anonymized}
    ph = Counter()
    words = 0
    for o in originals:
        a = by_id[o.get("encounter_id")]
        for f in TEXT_FIELDS:
            words += len(o[f].split())
            for m in re.findall(r"\[([A-Z_]+)(?:_\d+)?(?:[: ][^\]]*)?\]|\[(\d+[-+]\d*|<1)\]", a[f]):
                ph[m[0] or "AGE_BAND"] += 1
    return {"placeholders_per_1k_words": round(1000 * sum(ph.values()) / max(1, words), 2), "by_type": dict(ph.most_common())}
