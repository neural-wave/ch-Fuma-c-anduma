"""Context-aware detection of demographic quasi-identifiers that are not names:
body measurements (height, weight, BMI) and family size ("I have four children").

Only measurements of the person's body *stated as such* are detected (a cue such as "weighs", "height",
"BMI", "I am ... tall", a height-weight pair, a restatement in another unit); lesion sizes, doses (mg/kg),
lab values and weight *changes* ("lost 10 pounds") are never touched. Whether a record needs exact values
(children, weight-based dosing, dialysis, malnutrition, BMI ICD codes) is decided in the pipeline.
"""
from __future__ import annotations

import re

from .lexicon import NUM_WORDS_RE, words_to_number
from .spans import Span

I = re.I
_NUM = rf"(?:\d{{1,3}}(?:\.\d{{1,2}})?|{NUM_WORDS_RE})"
_WT_UNIT = r"(?:kg|kgs|kilograms?|kilos?|lbs?|pounds?)(?![/\w])"
# a weight cue before the value (same clause), and change words that make the number a difference
_WT_CUE = re.compile(r"\b(?:weigh(?:s|ed|ing|t|ts)?|wt|body\s+weight|scale\s+(?:says|shows|read)|i'?m|i\s+am|he\s+is|she\s+is|"
                     r"he's|she's|weighed\s+in\s+at)\b[^.;\n]{0,25}$", I)
_CHANGE = re.compile(r"\b(?:lost|lose|losing|gained?|gaining|loss|gain|dropped|drop|down|up|put\s+on|increase|decrease|"
                     r"by|of|about\s+\w+\s+(?:more|less)|another|extra|over|under)\s+(?:about\s+|around\s+|roughly\s+)?$", I)
_WEIGHT_RE = re.compile(rf"(?<![\w.])(?P<v>{_NUM})\s*(?P<u>{_WT_UNIT})", I)
_HT_CUE = re.compile(r"\b(?:height|ht|tall|stature|i'?m|i\s+am|he\s+is|she\s+is|he's|she's|measur\w*|stands?)\b[^.;\n]{0,20}$", I)
_HEIGHT_CM = re.compile(rf"(?<![\w.])(?P<v>{_NUM})\s*(?P<u>cm|centimet(?:er|re)s?|m|met(?:er|re)s?)(?![/\w²])", I)
_FT_IN = re.compile(r"(?<![\w.])(?P<ft>[3-7])\s*(?:'|’|ft\.?|feet|foot)\s*(?:(?:and\s+)?(?P<in>\d{1,2})\s*(?:\"|”|''|in\.?|inch(?:es)?)?)?"
                    r"(?![\w'])", I)
_FT_IN_WORDS = re.compile(r"\b(?P<ft>four|five|six|seven)\s+(?:feet|foot)(?:\s+(?:and\s+)?(?P<in>one|two|three|four|five|six|seven|"
                          r"eight|nine|ten|eleven)(?:\s+inch(?:es)?)?)?\b", I)
_TALL_AFTER = re.compile(r"^\s*(?:tall|in\s+height)\b", I)
_BMI_RE = re.compile(rf"\b(?:bmi|body\s+mass\s+index)\b\s*(?:of|is|was|at|=|:|today)?\s*(?:about\s+|around\s+)?(?P<v>{_NUM})", I)
_NOT_BODY = re.compile(r"\b(?:fundal|lesion|mass|wound|laceration|incision|nodule|cyst|tumou?r|ulcer|stone|polyp|"
                       r"circumference|diameter|length|width|depth|margin|gap|calf|thigh|arm|waist|neck|head)\b[^.;\n]{0,25}$", I)
# "recorded 160 cm and 70 kg": a height-weight pair; "weight 70 kg (reported elsewhere as 154 lb)": a restatement
_PAIR = re.compile(r"(?<![\w.])(?P<h>1\d{2}|2[0-2]\d)\s*cm\s*(?:and|,|/|with)\s*(?P<w>\d{2,3}(?:\.\d)?)\s*(?:kg|lbs?|pounds?)\b", I)
_RESTATE = re.compile(r"\s*(?:kg|kgs|kilograms?|kilos?|lbs?|pounds?)?\s*[(,;]?\s*(?:reported\s+(?:elsewhere\s+)?as|i\.e\.?|or|about|"
                      r"approx\w*\.?|equal\s+to|=|~)?\s*", I)
_SCALE_AFTER = re.compile(r"^\s*(?:on\s+(?:the|another|my|a|her|his|their)\s+(?:home\s+)?scale|at\s+(?:home|the\s+clinic)\s+on\s+the\s+scale)", I)
_FAMILY_RE = re.compile(rf"\b(?:(?P<subj>i|we|i've|we've|she|he|patient|the\s+patient)\s+(?:also\s+)?(?:have|has|had|raised)\s+(?:got\s+)?|"
                        rf"(?P<cue>there\s+(?:are|is)|household(?:\s+composition)?\s*[:,-]?|family\s+status\s*[:,-]?|"
                        rf"(?:married|divorced|widowed|separated|single|partnered|engaged)\s*[,;]?\s*(?:with|has|and|we\s+have)?|"
                        rf"(?:our|my|her|his|their)|"
                        rf"lives?\s+with(?:\s+(?:her|his|their|my|our))?|number\s+of\s+children\s*[:,-]?)\s+|"
                        rf"(?:mother|father|parent|mom|dad|grandmother|grandfather|grandparent)\s+(?:of|to)\s+)"
                        rf"(?P<v>no|{NUM_WORDS_RE}|\d{{1,2}})(?:\s+(?:grown|adult|young|little|small))?"
                        rf"(?P<noun>\s+(?:children|kids|sons|daughters|grandchildren|grandkids|stepchildren|boys|girls|"
                        rf"siblings|brothers|sisters|half-siblings|stepsiblings))?\b", I)
_HEREDITARY_AFTER = re.compile(r"^[^.;\n]{0,25}\b(?:with|affected|diagnosed|carriers?|who\s+(?:also\s+)?(?:have|has)|all\s+have)\b|"
                               r"^\s+(?:all\s+|also\s+|both\s+)?(?:have|has|had)\b[^.;\n]{0,30}\b(?:cancer|disease|diabetes|asthma|"
                               r"syndrome|disorder|history|hemophilia|mutation|gene|allerg\w*|\w+itis|\w+osis|\w+emia|\w+oma)\b", I)


def _sp(field, start, end, label, rule, text, reason, **meta) -> Span:
    return Span(field, start, end, label, f"rule:{rule}", 0.85, text[start:end], meta=meta, reasons=[reason])


def detect(text: str, field: str) -> list[Span]:
    out: list[Span] = []
    pairs = list(_PAIR.finditer(text))
    paired = {p.start("w") for p in pairs} | {p.start("h") for p in pairs}
    prev_wt_end = -100
    for m in _WEIGHT_RE.finditer(text):
        before = text[max(0, m.start() - 45):m.start()]
        v = words_to_number(m.group("v"))
        restated = m.start() - prev_wt_end < 45 and _RESTATE.fullmatch(text[prev_wt_end:m.start()])
        if v is None or _CHANGE.search(before) or not (_WT_CUE.search(before) or restated or m.start() in paired
                                                       or _SCALE_AFTER.match(text[m.end():])):
            continue
        unit = "lb" if m.group("u").lower().startswith(("lb", "pound")) else "kg"
        if (unit == "kg" and 2 <= v <= 300) or (unit == "lb" and 5 <= v <= 660):
            prev_wt_end = m.end()
            out.append(_sp(field, m.start("v"), m.end("v"), "ANTHROPOMETRIC", "weight", text, "body weight (cue, pair or restatement)",
                           kind="weight", unit=unit, value=v, form="digits" if m.group("v")[0].isdigit() else "words"))
    for m in _HEIGHT_CM.finditer(text):
        before = text[max(0, m.start() - 40):m.start()]
        v = words_to_number(m.group("v"))
        unit = "cm" if m.group("u").lower().startswith("c") else "m"
        ok = v is not None and ((unit == "cm" and 40 <= v <= 230) or (unit == "m" and 0.4 <= v <= 2.3))
        if ok and not _NOT_BODY.search(before) and (_HT_CUE.search(before) or _TALL_AFTER.match(text[m.end():])
                                                    or (unit == "cm" and m.start() in paired)):
            out.append(_sp(field, m.start("v"), m.end("v"), "ANTHROPOMETRIC", "height", text, "body height (cue or pair)",
                           kind="height", unit=unit, value=v, form="digits" if m.group("v")[0].isdigit() else "words"))
    for rx in (_FT_IN, _FT_IN_WORDS):
        for m in rx.finditer(text):
            before = text[max(0, m.start() - 40):m.start()]
            ft = words_to_number(m.group("ft"))
            inch = words_to_number(m.group("in")) if m.group("in") else 0
            quoted = rx is _FT_IN and m.group("in") and re.search(r"['’]", m.group(0))  # 5'7" notation
            if ft is None or inch is None or inch > 11 or _NOT_BODY.search(before):
                continue
            if quoted or _HT_CUE.search(before) or _TALL_AFTER.match(text[m.end():]):
                out.append(_sp(field, m.start(), m.end(), "ANTHROPOMETRIC", "height", text, "body height in feet/inches",
                               kind="height", unit="ftin", value=ft * 12 + inch))
    for m in _BMI_RE.finditer(text):
        v = words_to_number(m.group("v"))
        if v is not None and 10 <= v <= 90:
            out.append(_sp(field, m.start("v"), m.end("v"), "ANTHROPOMETRIC", "bmi", text, "body mass index value",
                           kind="bmi", value=v, form="digits" if m.group("v")[0].isdigit() else "words"))
    for m in _FAMILY_RE.finditer(text):
        n = 0 if m.group("v").lower() == "no" else words_to_number(m.group("v"))
        cue = m.group("cue") or ""
        if n is None or not (0 <= n <= 20) or ((m.group("subj") or cue) and not m.group("noun")
                                              and not cue.lower().startswith("number")):
            continue  # "I have two concerns", "there are two options": the count must be of children
        if _HEREDITARY_AFTER.match(text[m.end():]):
            continue  # "three children with hemophilia": an affected count is family history, kept exactly
        out.append(_sp(field, m.start("v"), m.end("v"), "FAMILY_SIZE", "family-size", text, "number of the patient's children",
                       count=n))
    return out
