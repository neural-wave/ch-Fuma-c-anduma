"""Attacker feature extraction. Inputs: the ANONYMISED release (MedAnon output JSONL) and the attacker's own
register. Nothing from the generator, the original text or the truth is imported or read here.

Release features are parsed from the published text only. Generalised values stay intervals: an age band is never
read as an exact age, and a calendar year is a birth year only when it is published as one ("[DOB] 1978").
Birth year is the common currency: age a published at visit year v means birth year in [v-a-1, v-a]; when the visit
year is not published, the attacker uses the release window (2023-2026, assumed public knowledge).
Dictionary searches (names, cities, occupation titles, hobbies) use only values present in the attacker's register.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REF_YEAR = 2026            # register snapshot (attacker side)
WINDOW = (2023, 2026)      # assumed public: collection period of the released encounters
MEDANON_LEXICON = Path(__file__).resolve().parents[4] / "src" / "medanon" / "resources"  # public resources

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
NUMW = {w: i for i, w in enumerate(_ONES)} | {w: 10 * i for i, w in enumerate(_TENS) if w != "_"}
_NUMW_RE = r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?" \
           r"|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|" \
           r"seventeen|eighteen|nineteen|a hundred)"
MONTHS = {m.lower(): i for i, m in enumerate("January February March April May June July August September October "
                                             "November December".split(), 1)}


def num(s: str) -> int | None:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    if s == "a hundred":
        return 100
    parts = re.split(r"[ -]", s)
    try:
        return sum(NUMW[p] for p in parts)
    except KeyError:
        return None


def spoken_year(s: str) -> int | None:
    w = s.lower().split()
    if len(w) >= 2 and w[0] == "two" and w[1] == "thousand":
        return 2000 + (num(" ".join(w[2:])) or 0) if len(w) > 2 else 2000
    if len(w) >= 2 and w[0] in ("nineteen", "twenty"):
        rest = " ".join(x for x in w[1:] if x != "oh")
        r = num(rest)
        return None if r is None else (1900 if w[0] == "nineteen" else 2000) + r
    return None


def _year_any(s: str, ref: int) -> int | None:
    """4-digit, 2-digit (century by recency) or spoken year."""
    s = s.strip()
    if re.fullmatch(r"\d{4}", s):
        return int(s)
    if re.fullmatch(r"\d{2}", s):
        y = int(s)
        return 2000 + y if y <= ref % 100 else 1900 + y
    return spoken_year(s)


# ------------------------------------------------------------------ attacker-side category map (MedAnon public lexicon)
def load_occupation_categories() -> dict[str, str]:
    out = {}
    p = MEDANON_LEXICON / "occupations.tsv"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "\t" in line:
                term, cat = line.split("\t")[:2]
                out[term.strip().lower()] = cat.strip().lower()
    return out


def occ_category(title: str | None, cats: dict[str, str]) -> str | None:
    if not isinstance(title, str) or not title:
        return None
    t = title.lower()
    if t in cats:
        return cats[t]
    for term in sorted(cats, key=len, reverse=True):  # head-word match: "line cook" -> "cook"
        if re.search(rf"\b{re.escape(term)}\b", t):
            return cats[term]
    return None


def load_common_words(n: int = 5000) -> set[str]:
    p = MEDANON_LEXICON / "common_words.tsv"
    words = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                words.append(line.split("\t")[0].strip().lower())
    return set(words[:n])


# ------------------------------------------------------------------ release features
_AGE_EXACT = re.compile(r"(?<![\[\d.\-])(\d{1,3})(?:-| )?(?:year|yr)s?(?:-| )old|(?<![\[\d.])(\d{1,3}) ?y/?o\b", re.I)
_AGE_BAND = re.compile(r"\[(\d{1,2})-(\d{1,3})\](?:-| )?(?:year|yr|y/?o)", re.I)
_AGE_TOP = re.compile(r"(?<!\d)(\d{2,3})\+(?:\])?(?:-| )?(?:year|yr|y/?o)", re.I)
_AGE_WORDS = re.compile(rf"\b(?:i'?m|i am|just turned|patient is an?)\s+({_NUMW_RE})(?:\s+years?\s+old)?\b(?!\s*(?:minutes|weeks|days|months|times|kids|children))", re.I)
_AGE_WORDS_YO = re.compile(rf"\b({_NUMW_RE})\s+years?\s+old\b", re.I)
_AGE_BAND_SPOKEN = re.compile(r"\b(?:i'?m|i am|just turned|is an?)\s+\[(\d{1,2})-(\d{1,3})\]", re.I)
_DOB_YEAR = re.compile(r"\[DOB\]\s*((?:19|20)\d{2}\b|\d{2}\b|(?:nineteen|twenty|two thousand)(?: [a-z]+){0,2})")
_AGE_Q = re.compile(r"\[doctor\][^\n]*\bhow old\b[^\n]*\n\[patient\]\s*(?:(?:um|uh|so|well|okay|like|you know)\s*,\s*)*"
                    rf"(?:i'?m\s+|i am\s+|i just turned\s+)?(?:\[(\d{{1,2}})-(\d{{1,3}})\]|(\d{{1,3}})\b|({_NUMW_RE})\b)", re.I)
_DOB_FULL = re.compile(r"\b(?:DOB|date of birth)\s*:?\s*(?:(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})|([A-Z][a-z]+) (\d{1,2}), ((?:19|20)\d{2}))", re.I)
_VISIT_LINE = re.compile(r"^(?![^\n]*(?:birth|DOB))[A-Za-z ]{0,25}\bdate\b[A-Za-z ]{0,15}:\s*(.*)$", re.I | re.M)
_YEAR = re.compile(r"(?<![\d/.\-])((?:19[3-9]|20[0-2])\d)(?![\d/])")
_FULLDATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b|\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b|\b([A-Z][a-z]+) (\d{1,2}), ((?:19|20)\d{2})\b")
_OCC_CAT = re.compile(r"\[OCCUPATION:\s*([a-z ]+)\]")
_SEX_M = re.compile(r"\b(?:he|his|him|male|man|gentleman|mr\.?|boy|young man|father of)\b", re.I)
_SEX_F = re.compile(r"\b(?:she|her|hers|female|woman|lady|mrs\.?|ms\.?|girl|young woman|mother of)\b", re.I)
_RACE = re.compile(r"\b(White|Black|Asian|Hispanic|African American|Caucasian|Latino|Latina)\b(?=[^.\n]{0,25}\b(?:male|female|man|woman|boy|girl))")
_RACE_MAP = {"white": "white", "caucasian": "white", "black": "black", "african american": "black", "asian": "asian",
             "hispanic": "hispanic", "latino": "hispanic", "latina": "hispanic"}
_MARITAL = [("W", re.compile(r"\bwidow(?:ed|er)?\b|\bmy late (?:husband|wife)\b", re.I)),
            ("D", re.compile(r"\bdivorced\b|\bmy ex\b", re.I)),
            ("M", re.compile(r"\bmarried\b|\bmy (?:husband|wife|spouse)\b", re.I)),
            ("S", re.compile(r"\bsingle\b|\bnever married\b", re.I))]
_KIDS = re.compile(rf"(?:\b(\d{{1,2}})|\b({_NUMW_RE})|\b(no)|\[(\d)\+\])\s+(?:grown\s+|adult\s+|young\s+)?(?:children|kids)\b|"
                   rf"\b(?:father|mother) of (?:(\d{{1,2}})|\[(\d)\+\])", re.I)
_WORK = [("night shifts", re.compile(r"\bnight shifts?\b|\bwork(?:s|ing)? nights\b", re.I)),
         ("rotating shifts", re.compile(r"\brotating shifts\b|\bshifts rotate\b", re.I)),
         ("remote", re.compile(r"\bwork(?:s|ing)? (?:from home|remotely)\b", re.I)),
         ("long hauls", re.compile(r"\bon the road\b|\blong[- ]haul", re.I)),
         ("part-time", re.compile(r"\bpart[ -]time\b", re.I))]
_IV = r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]"
_HT = [(re.compile(rf"\bheight\s*:?\s*(?:{_IV}|(\d{{2,3}}(?:\.\d)?))\s*cm\b", re.I), 1.0),
       (re.compile(r"\bheight\s*:?\s*(\d)'(\d{1,2})\"", re.I), "ftin"),
       (re.compile(r"\bheight\s*:?\s*\[(\d) ft (\d{1,2}) in - (\d) ft (\d{1,2}) in\]", re.I), "ftin_iv"),
       (re.compile(rf"\b(?:tall|height)\b[^.\n]{{0,20}}?(?:{_IV}|(\d{{3}}))\s*(?:centimeters|cm)\b", re.I), 1.0)]
_WT = [(re.compile(rf"\bweight\s*:?\s*(?:{_IV}|(\d{{1,3}}(?:\.\d)?))\s*(kg|lbs?)\b", re.I), None),
       (re.compile(rf"\b(?:weigh\w*|scale)\b[^.\n]{{0,30}}?(?:{_IV}|(\d{{1,3}}(?:\.\d)?))\s*(kilos|kilograms|kgs?|pounds|lbs?)\b", re.I), None)]
_BMI = re.compile(r"\bBMI\s*:?\s*(?:\[([<>=]*)(\d+(?:\.\d)?)(?:-(\d+(?:\.\d)?))?\]|(\d{2}(?:\.\d)?))", re.I)
_CITY_CUE = re.compile(r"(?:\bin|\bfrom|\bto|\bnear|\blives in|\bmoved to|\bout in|\bover in)\s+$", re.I)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+")


class Vocab:
    """Attacker-side dictionaries built from the register only."""

    def __init__(self, reg: pd.DataFrame, cats: dict[str, str], common: set[str]):
        self.cats = cats
        names = set(reg["first"].str.lower()) | set(reg["last"].str.lower())
        self.names = {n for n in names if len(n) > 2 and n not in common}
        self.cities = sorted({c.lower() for c in reg.get("city", pd.Series(dtype=str)).dropna()}, key=len, reverse=True)
        self.titles = sorted({c.lower() for c in reg.get("occupation", pd.Series(dtype=str)).dropna()}, key=len, reverse=True)
        self.hobbies = sorted({c.lower() for c in reg.get("hobby", pd.Series(dtype=str)).dropna()}, key=len, reverse=True)
        self._city_re = re.compile(r"\b(" + "|".join(map(re.escape, self.cities)) + r")\b", re.I) if self.cities else None
        self._title_re = re.compile(r"\b(" + "|".join(map(re.escape, self.titles)) + r")\b", re.I) if self.titles else None
        self._hobby_re = re.compile(r"\b(" + "|".join(map(re.escape, self.hobbies)) + r")\b", re.I) if self.hobbies else None


def _interval(m_lo, m_hi, m_exact, scale=1.0, half=0.5):
    if m_lo is not None:
        return float(m_lo) * scale, float(m_hi) * scale
    if m_exact is not None:
        v = float(m_exact) * scale
        return v - half * scale, v + half * scale
    return None


def extract(rec: dict, voc: Vocab) -> dict:
    note, dia = rec.get("note", ""), rec.get("dialogue", "")
    text = note + "\n" + dia
    f: dict = {"rid": rec["encounter_id"]}
    # ---- visit year (administrative header only) and residual full visit date
    vy, vym = None, None
    for m in _VISIT_LINE.finditer(note):
        line = m.group(1)
        if fd := _FULLDATE.search(line):
            g = fd.groups()
            y, mo = (int(g[2]), int(g[0])) if g[2] else (int(g[3]), int(g[4])) if g[3] else (int(g[8]), MONTHS.get(g[6].lower(), 0))
            vy, vym = y, f"{y}-{mo:02d}" if mo else None
        elif ym := _YEAR.search(line):
            vy = int(ym.group(1))
        break
    f["vis_y"] = [vy] if vy else None
    f["vis_ym"] = [vym] if vym else None
    v_lo, v_hi = (vy, vy) if vy else WINDOW
    # ---- age -> birth-year interval
    found = []  # one age interval per field; the attacker intersects consistent ones (most precise wins)
    for src in (note, dia):
        iv = None
        if m := _AGE_BAND.search(src):
            iv = (int(m.group(1)), int(m.group(2)))
        elif m := _AGE_TOP.search(src):
            iv = (int(m.group(1)), 120)
        elif m := _AGE_EXACT.search(src):
            a = int(m.group(1) or m.group(2))
            iv = (a, a) if a < 120 else None
        elif (m := _AGE_BAND_SPOKEN.search(src)):
            iv = (int(m.group(1)), int(m.group(2)))
        if (iv is None or iv[0] != iv[1]) and src is dia and (m := _AGE_Q.search(src)):
            a = num(m.group(3) or m.group(4)) if not m.group(1) else None
            q = (int(m.group(1)), int(m.group(2))) if m.group(1) else ((a, a) if a is not None and 0 < a < 120 else None)
            iv = q if iv is None or (q and iv[0] <= q[0] and q[1] <= iv[1]) else iv
        if iv is None and ((m := _AGE_WORDS_YO.search(src)) or (m := _AGE_WORDS.search(src))):
            a = num(m.group(1))
            iv = (a, a) if a is not None and 0 < a < 120 else None
        if iv:
            found.append(iv)
    age_iv = found[0] if found else None
    for iv in found[1:]:
        if iv[0] <= age_iv[1] and age_iv[0] <= iv[1]:
            age_iv = (max(iv[0], age_iv[0]), min(iv[1], age_iv[1]))
    by = None
    if age_iv:
        by = (v_lo - age_iv[1] - 1, v_hi - age_iv[0])
    dob_y = None
    if m := _DOB_YEAR.search(text):
        dob_y = _year_any(m.group(1), v_hi)
    md = None
    if m := _DOB_FULL.search(text):
        g = m.groups()
        if g[2]:
            dob_y, md = int(g[2]), f"{int(g[0]):02d}-{int(g[1]):02d}"
        elif g[5] and g[3].lower() in MONTHS:
            dob_y, md = int(g[5]), f"{MONTHS[g[3].lower()]:02d}-{int(g[4]):02d}"
    if dob_y is None:  # spoken date of birth in the dialogue (answer after a date-of-birth question)
        m = re.search(r"\[doctor\][^\n]*(?:date of birth|birthday)[^\n]*\n\[patient\]\s*([^\n]+)", dia, re.I)
        if m:
            ans = m.group(1)
            if (ym := re.search(r"\[(?:DOB|DATE)\]\s*((?:19|20)\d{2}\b|\d{2}\b|(?:nineteen|twenty|two thousand)(?: [a-z]+){0,2})", ans)):
                dob_y = _year_any(ym.group(1), v_hi)
            elif dm := re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", ans):
                yy = int(dm.group(3))
                dob_y = yy if yy > 999 else (1900 + yy if yy > v_hi % 100 else 2000 + yy)
                md = f"{int(dm.group(1)):02d}-{int(dm.group(2)):02d}"
            elif sm := re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
                                 r"[^,\n]*,\s*((?:nineteen|twenty|two thousand)[a-z ]*)", ans, re.I):
                dob_y = spoken_year(sm.group(2).strip())
    if dob_y:
        by = (dob_y, dob_y) if by is None or by[0] <= dob_y <= by[1] else by  # conflicting sources: keep the age
    f["by_lo"], f["by_hi"] = (by if by else (None, None))
    f["dob_md"] = md
    # ---- event years: every published calendar year except birth-year and visit-header lines
    ev = set()
    for line in text.splitlines():
        if re.search(r"\[DOB\]|\bDOB\b|date of birth", line, re.I) or _VISIT_LINE.match(line):
            continue
        ev |= {int(y) for y in _YEAR.findall(line)}
    ev |= {y for y in (spoken_year(s) for s in re.findall(r"\b(?:nineteen|twenty|two thousand)(?: [a-z]+){1,2}", dia)) if y}
    ev.discard(vy)
    f["ev"] = sorted(y for y in ev if 1930 <= y <= REF_YEAR) or None
    # ---- sex
    nm, nf = len(_SEX_M.findall(note)), len(_SEX_F.findall(note))
    f["sex"] = "M" if nm > 1.5 * nf and nm else "F" if nf > 1.5 * nm and nf else None
    # ---- race (published ethnicity words next to the demographic noun)
    m = _RACE.search(note)
    f["race"] = _RACE_MAP[m.group(1).lower()] if m else None
    # ---- occupation: residual title (register dictionary) or published category
    title = None
    if voc._title_re and (m := voc._title_re.search(text)):
        title = m.group(1).lower()
    cat = None
    if m := _OCC_CAT.search(text):
        cat = m.group(1).strip()
    f["occ_title"] = title
    f["occ_cat"] = cat or occ_category(title, voc.cats)
    # ---- marital, children, work pattern, hobby
    f["marital"] = next((k for k, rx in _MARITAL if rx.search(text)), None)
    kl = kh = None
    for m in _KIDS.finditer(text):
        g = m.groups()
        if g[0] or g[4]:
            kl = kh = int(g[0] or g[4])
        elif g[1]:
            n = num(g[1])
            kl = kh = n
        elif g[2]:
            kl = kh = 0
        elif g[3] or g[5]:
            kl, kh = int(g[3] or g[5]), 20
        if kl is not None:
            break
    f["kids_lo"], f["kids_hi"] = kl, kh
    f["work"] = next((k for k, rx in _WORK if rx.search(text)), None)
    f["hb"] = sorted({m.group(1).lower() for m in voc._hobby_re.finditer(text)}) or None if voc._hobby_re else None
    # ---- body measurements (exact -> +-0.5 unit; published interval kept as interval)
    ht = None
    for rx, mode in _HT:
        if m := rx.search(text):
            g = m.groups()
            if mode == "ftin":
                c = (int(g[0]) * 12 + int(g[1])) * 2.54
                ht = (c - 1.3, c + 1.3)
            elif mode == "ftin_iv":
                ht = ((int(g[0]) * 12 + int(g[1])) * 2.54, (int(g[2]) * 12 + int(g[3])) * 2.54)
            else:
                ht = _interval(g[0], g[1], g[2])
            break
    wt = None
    for rx, _ in _WT:
        if m := rx.search(text):
            g = m.groups()
            scale = 1 / 2.2046 if g[3].lower().startswith(("lb", "pound")) else 1.0
            wt = _interval(g[0], g[1], g[2], scale=scale)
            break
    bmi = None
    if m := _BMI.search(text):
        op, a, b, ex = m.groups()
        if ex:
            bmi = (float(ex) - 0.05, float(ex) + 0.05)
        elif op and op.startswith("<"):
            bmi = (10.0, float(a))
        elif op and op.startswith(">"):
            bmi = (float(a), 90.0)
        elif a and b:
            bmi = (float(a), float(b) + 0.1)
    for k, iv in (("ht", ht), ("wt", wt), ("bmi", bmi)):
        f[f"{k}_lo"], f[f"{k}_hi"] = (round(iv[0], 2), round(iv[1], 2)) if iv else (None, None)
    # ---- residual names and city (register dictionary; never from placeholders)
    toks = {t.lower().removesuffix("'s") for t in _TOKEN.findall(re.sub(r"\[[^\]]*\]", " ", text))}
    f["nm_found"] = sorted(toks & voc.names) or None
    city = None
    if voc._city_re:
        for m in voc._city_re.finditer(text):
            if _CITY_CUE.search(text[max(0, m.start() - 12):m.start()]):
                city = m.group(1).lower()
                break
    f["city"] = city
    # ---- ICD-10 codes (released unchanged by design)
    f["icd"] = sorted({c["code"][:3] for c in rec.get("icd10_codes", [])}) or None
    return f


def register_features(reg: pd.DataFrame, cats: dict[str, str]) -> pd.DataFrame:
    """Register rows -> the same feature schema as the release (intervals for uncertain knowledge)."""
    g = lambda c: reg[c] if c in reg else pd.Series([None] * len(reg))  # noqa: E731
    out = pd.DataFrame({"key": reg["key"]})
    out["nm"] = [[str(a).lower(), str(b).lower()] for a, b in zip(reg["first"], reg["last"])]
    out["sex"] = g("sex")
    bd, byr, aa, tol = g("birth_date"), g("birth_year"), g("age_approx"), g("age_tol")
    lo, hi, md = [], [], []
    for d, y, a, t in zip(bd, byr, aa, tol):
        if isinstance(d, str):
            lo.append(int(d[:4])), hi.append(int(d[:4])), md.append(d[5:10])
        elif y == y and y is not None:
            lo.append(int(y)), hi.append(int(y)), md.append(None)
        elif a == a and a is not None:
            t = t if t == t and t is not None else 3  # tolerance unknown -> assume +-3 years
            lo.append(int(REF_YEAR - a - t - 1)), hi.append(int(REF_YEAR - a + t)), md.append(None)
        else:
            lo.append(None), hi.append(None), md.append(None)
    out["by_lo"], out["by_hi"], out["dob_md"] = lo, hi, md
    out["city"] = g("city").str.lower() if "city" in reg else None
    out["occ_title"] = g("occupation").str.lower() if "occupation" in reg else None
    memo = {t: occ_category(t, cats) for t in set(out["occ_title"].dropna())} if "occupation" in reg else {}
    out["occ_cat"] = [memo.get(t) for t in out["occ_title"]] if "occupation" in reg else None
    out["marital"] = g("marital")
    out["kids_lo"] = out["kids_hi"] = g("children")
    out["work"] = g("work_pattern")
    out["hb"] = [[h.lower()] if isinstance(h, str) else None for h in g("hobby")]
    out["race"] = g("race")
    out["ht_lo"], out["ht_hi"] = g("height_lo"), g("height_hi")
    out["wt_lo"], out["wt_hi"] = g("weight_lo"), g("weight_hi")
    bl, bh = [], []
    for hl, hh, wl, wh in zip(out["ht_lo"], out["ht_hi"], out["wt_lo"], out["wt_hi"]):
        ok = all(x == x and x is not None for x in (hl, hh, wl, wh))
        bl.append(round(wl / (hh / 100) ** 2, 2) if ok else None)
        bh.append(round(wh / (hl / 100) ** 2, 2) if ok else None)
    out["bmi_lo"], out["bmi_hi"] = bl, bh
    out["ev"] = [list(map(int, e)) if e is not None and len(e) else None for e in g("event_years")]
    out["icd"] = [list(e) if e is not None and len(e) else None for e in g("icd3")]
    vym = [list(v) if v is not None and len(v) else None for v in g("visit_ym")]
    out["vis_ym"] = vym
    out["vis_y"] = [sorted({int(x[:4]) for x in v}) if v else None for v in vym]
    return out
