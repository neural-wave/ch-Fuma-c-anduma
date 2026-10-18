"""Synthetic gold-standard encounters for measuring privacy recall and medical precision.

The real data has no gold annotations (and must not be inspected), so we generate encounters in
the same two styles - lowercase ASR dialogue with [doctor]/[patient] tags and a cased SOAP note -
with exactly known PHI spans. Templates deliberately include hard cases: names that are common
words (will, grace, hope), names without any cue, surnames that are eponyms (Mr. Parkinson WITH
Parkinson's disease), drug brands that look like names, spoken numbers and dates. "Trap" spans
are medical expressions that must survive anonymisation untouched.

A second style ("discharge") mimics hospital discharge summaries with phone numbers, addresses,
e-mails and attending physicians, for the robustness bonus task.
"""
from __future__ import annotations

import random
import re

from ..lexicon import number_to_words

FIRST = ["andrew", "maria", "james", "linda", "robert", "jennifer", "carlos", "aisha", "wei", "priya", "olga",
         "dmitri", "fatima", "juan", "keisha", "tyrone", "connor", "brittany", "hiroshi", "mohammed", "ingrid",
         "sven", "giulia", "matteo", "deshawn", "siobhan", "ravi", "elena", "kofi", "beatriz", "lucas", "hannah"]
AMBIG_FIRST = ["will", "grace", "hope", "mark", "rose", "april", "june", "jordan", "faith", "joy", "bill",
               "frank", "sandy", "dawn", "summer", "hunter", "chase", "mason", "gene", "ray"]
LAST = ["perez", "kowalski", "nguyen", "okafor", "schmidt", "rossi", "yamamoto", "fitzgerald", "martinez",
        "lindqvist", "haddad", "dubois", "novak", "petrov", "brennan", "castillo", "abernathy", "whitfield",
        "oyelaran", "takahashi", "mcallister", "delgado", "johansson", "kaminski", "osei", "romero"]
EPONYM_LAST = {"parkinson": ("parkinson's disease", "G20"), "bell": ("bell's palsy", "G51.0"),
               "graves": ("graves' disease", "E05.00"), "baker": ("baker's cyst", "M71.20"),
               "turner": ("turner syndrome", "Q96.9"), "wilson": ("wilson's disease", "E83.01"),
               "addison": ("addison's disease", "E27.1"), "barrett": ("barrett's esophagus", "K22.70"),
               "cushing": ("cushing's syndrome", "E24.9"), "crohn": ("crohn's disease", "K50.90")}
CONDITIONS = [("type 2 diabetes", "E11.9"), ("hypertension", "I10"), ("lyme disease", "A69.20"),
              ("hashimoto's thyroiditis", "E06.3"), ("alzheimer's disease", "G30.9"),
              ("hodgkin lymphoma", "C81.90"), ("achilles tendinitis", "M76.60"), ("gout", "M10.9"),
              ("atrial fibrillation", "I48.91"), ("copd", "J44.9"), ("major depressive disorder", "F32.A"),
              ("osteoarthritis of the knee", "M17.9"), ("kidney stones", "N20.0"), ("gerd", "K21.9")]
DRUGS = ["lasix", "flonase", "allegra", "eliquis", "xarelto", "januvia", "jardiance", "ozempic", "humira",
         "lipitor", "metformin", "lisinopril", "tylenol", "advil", "ambien", "lyrica", "celebrex", "keppra",
         "zoloft", "synthroid", "norvasc", "plavix", "singulair", "prednisone"]
CITIES = ["springfield", "tucson", "albuquerque", "savannah", "boise", "spokane", "fresno", "duluth",
          "scranton", "flagstaff", "chattanooga", "kalamazoo", "pensacola", "bakersfield"]
STATES = ["ohio", "georgia", "vermont", "arizona", "oregon", "kentucky", "nevada", "maine", "texas", "iowa"]
COUNTRIES = ["mexico", "brazil", "kenya", "vietnam", "india", "portugal", "peru", "ghana", "thailand"]
EMPLOYERS = ["fedex", "home depot", "costco", "boeing", "walmart", "target", "ups", "kroger", "lockheed martin",
             "starbucks", "verizon", "pfizer"]
HOSPITALS = ["mercy hospital", "st. luke's medical center", "riverside clinic", "good samaritan hospital",
             "memorial hermann hospital"]
SCHOOLS = ["lincoln elementary", "ohio state university", "jefferson high school", "rutgers university"]
OCCUPATIONS = ["teacher", "truck driver", "nurse", "electrician", "accountant", "firefighter", "cashier",
               "software engineer", "plumber", "police officer", "chef", "pharmacist", "mechanic", "farmer"]
RELATIONS = ["wife", "husband", "daughter", "son", "sister", "brother", "neighbor", "mother"]
HOLIDAYS = ["christmas", "thanksgiving", "easter", "the fourth of july", "new year's"]
MONTHS = ["january", "february", "march", "april", "june", "july", "august", "september", "october", "november"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
FILLERS = [
    "[doctor] okay . any chest pain or shortness of breath ?\n[patient] no , not really .",
    "[doctor] and how have you been sleeping ?\n[patient] um , not great , i wake up a lot at night .",
    "[doctor] any fevers or chills ?\n[patient] no .",
    "[doctor] let me take a listen to your heart and lungs . take a deep breath for me .",
    "[patient] i've been trying to walk more , uh , maybe twenty minutes a day .",
    "[doctor] your blood pressure today is one thirty two over eighty four , which is okay .",
    "[doctor] i'm going to order some labs , a cbc and a cmp , and we'll go from there .",
    "[patient] the pain is mostly in my lower back and it goes down my left leg .",
    "[doctor] on exam , there's some tenderness over the lateral malleolus but no swelling .",
]


class Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[dict] = []
        self.traps: list[dict] = []
        self.len = 0

    def add(self, text: str) -> "Builder":
        self.parts.append(text)
        self.len += len(text)
        return self

    def phi(self, text: str, label: str) -> "Builder":
        self.spans.append({"start": self.len, "end": self.len + len(text), "label": label, "text": text})
        return self.add(text)

    def trap(self, text: str) -> "Builder":
        self.traps.append({"start": self.len, "end": self.len + len(text), "text": text})
        return self.add(text)

    def tpl(self, template: str, slots: dict) -> "Builder":
        """Fill '{slot}' markers; slots map name -> (text, label|None|'TRAP')."""
        for piece in re.split(r"(\{\w+\})", template):
            m = re.fullmatch(r"\{(\w+)\}", piece)
            if not m:
                self.add(piece)
                continue
            text, label = slots[m.group(1)]
            if label == "TRAP":
                self.trap(text)
            elif label:
                self.phi(text, label)
            else:
                self.add(text)
        return self


def _cap(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def _persona(rng: random.Random) -> dict:
    eponym = rng.random() < 0.3
    last = rng.choice(list(EPONYM_LAST)) if eponym else rng.choice(LAST)
    first = rng.choice(AMBIG_FIRST) if rng.random() < 0.3 else rng.choice(FIRST)
    cond, code = EPONYM_LAST[last] if eponym else rng.choice(CONDITIONS)
    cond2, code2 = rng.choice(CONDITIONS)
    sex = rng.choice(["male", "female"])
    year = rng.randint(1940, 2000)
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    return {
        "first": first, "last": last, "sex": sex, "he": "he" if sex == "male" else "she",
        "his": "his" if sex == "male" else "her", "age": 2024 - year, "dob": (month, day, year),
        "mrn": str(rng.randint(100000, 9999999)), "city": rng.choice(CITIES), "state": rng.choice(STATES),
        "country": rng.choice(COUNTRIES), "employer": rng.choice(EMPLOYERS), "hospital": rng.choice(HOSPITALS),
        "school": rng.choice(SCHOOLS), "occupation": rng.choice(OCCUPATIONS), "relation": rng.choice(RELATIONS),
        "fam": rng.choice(FIRST + AMBIG_FIRST), "doc": rng.choice(LAST + ["shah", "lee", "patel", "cohen"]),
        "cond": cond, "code": code, "cond2": cond2, "code2": code2, "drug": rng.choice(DRUGS),
        "drug2": rng.choice(DRUGS), "a1c": round(rng.uniform(5.8, 11.5), 1), "holiday": rng.choice(HOLIDAYS),
        "month": rng.choice(MONTHS), "weekday": rng.choice(WEEKDAYS), "year": str(rng.randint(2005, 2022)),
        "visit": (rng.choice(MONTHS), rng.randint(1, 28)), "pet": rng.choice(["max", "buddy", "bella", "luna", "rocky"]),
    }


def _dob_spoken(p: dict, rng: random.Random) -> str:
    m, d, y = p["dob"]
    fmt = rng.choice(["slash", "comma", "words"])
    if fmt == "slash":
        return f"{m}/{d}/{y}"
    if fmt == "comma":
        return f"{m} , {d} , {y}"
    month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december"][m - 1]
    return f"{month} {d}th , {y}"


def _age_text(p: dict, rng: random.Random) -> str:
    return number_to_words(p["age"], 0) if rng.random() < 0.3 else str(p["age"])


def dialogue(p: dict, rng: random.Random) -> Builder:
    b = Builder()
    full = f"{p['first']} {p['last']}"
    opener = rng.choice(["virtscribe", "virtassist", "aci"])
    if opener == "virtscribe":
        b.add("[doctor] ").phi(full, "PERSON").add(" , date of birth ").phi(_dob_spoken(p, rng), "DOB") \
            .add(" , mrn ").phi(p["mrn"], "ID").add(" . ").tpl("{he} is a {age} year old {sex} here for {cond} .\n",
                                                               {"he": (p["he"], None), "age": (_age_text(p, rng), "AGE"), "sex": (p["sex"], None), "cond": (p["cond"], "TRAP")})
    elif opener == "virtassist":
        b.tpl("[doctor] hey dragon , i'm seeing {full} today . {he} is a {age}-year-old {sex} with {cond} .\n",
              {"full": (full, "PERSON"), "he": (p["he"], None), "age": (str(p["age"]), "AGE"), "sex": (p["sex"], None), "cond": (p["cond"], "TRAP")})
    b.tpl(rng.choice(["[doctor] hi {first} , how are you doing today ?\n", "[doctor] good morning , mr. {last} . how are you ?\n",
                      "[doctor] hello ms. {last} , nice to see you .\n", "[doctor] hey {first} , good to see you again .\n"]),
          {"first": (p["first"], "PERSON"), "last": (p["last"], "PERSON")})
    b.tpl(rng.choice(["[patient] hi doctor {doc} . i'm okay .\n", "[patient] i'm doing alright , dr. {doc} .\n",
                      "[patient] good , thanks .\n"]), {"doc": (p["doc"], "PERSON")})
    turns = [
        ("[patient] my {rel} , {fam} , has been helping me with the {drug} .\n", {"rel": (p["relation"], None), "fam": (p["fam"], "PERSON"), "drug": (p["drug"], "TRAP")}),
        ("[patient] {fam} is going to pick me up after this .\n", {"fam": (p["fam"], "PERSON")}),
        ("[patient] i work as a {occ} at {emp} , so i'm on my feet all day .\n", {"occ": (p["occupation"], "OCCUPATION"), "emp": (p["employer"], "ORGANIZATION")}),
        ("[patient] i'm a {occ} , been doing it for twenty years .\n", {"occ": (p["occupation"], "OCCUPATION")}),
        ("[patient] {emp} has been good about giving me time off .\n", {"emp": (p["employer"], "ORGANIZATION")}),
        ("[patient] we just got back from a trip to {country} .\n", {"country": (p["country"], "LOCATION")}),
        ("[patient] i grew up in {state} but we moved to {city} last year .\n", {"state": (p["state"], "LOCATION"), "city": (p["city"], "LOCATION")}),
        ("[patient] {city} has been so hot this summer .\n", {"city": (p["city"], "LOCATION")}),
        ("[patient] i saw doctor {doc2} over at {hosp} back in {month} .\n", {"doc2": (rng.choice(LAST), "PERSON"), "hosp": (p["hospital"], "ORGANIZATION"), "month": (p["month"], "MONTH")}),
        ("[patient] it all started around {hol} .\n", {"hol": (p["holiday"], "HOLIDAY")}),
        ("[patient] in {year} i had my gallbladder taken out .\n", {"year": (p["year"], "YEAR")}),
        ("[patient] my granddaughter goes to {school} .\n", {"school": (p["school"], "ORGANIZATION")}),
        ("[patient] my dog {pet} keeps me active .\n", {"pet": (p["pet"], "PERSON")}),
        ("[patient] i turned {age} in {month} .\n", {"age": (_age_text(p, rng), "AGE"), "month": (p["month"], "MONTH")}),
        ("[doctor] your {cond} looks stable . keep taking the {drug2} .\n", {"cond": (p["cond"], "TRAP"), "drug2": (p["drug2"], "TRAP")}),
        ("[doctor] and how is the {cond2} ? are you still on {drug} ?\n", {"cond2": (p["cond2"], "TRAP"), "drug": (p["drug"], "TRAP")}),
        ("[doctor] your a1c came back at {a1c} .\n", {"a1c": (number_to_words(p["a1c"], 1) if rng.random() < 0.5 else str(p["a1c"]), None)}),
        ("[doctor] will you be able to come back on {wd} ?\n[patient] yes , that works .\n", {"wd": (p["weekday"], "WEEKDAY")}),
        ("[doctor] hey dragon , show me the {test} results .\n", {"test": (rng.choice(["chest x-ray", "mri", "echocardiogram", "ekg"]), "TRAP")}),
    ]
    for t, slots in rng.sample(turns, k=rng.randint(9, 14)):
        b.tpl(t, slots)
        if rng.random() < 0.4:
            b.add(rng.choice(FILLERS) + "\n")
    m, d = p["visit"]
    title = rng.choice(["", "mr. " if p["sex"] == "male" else "ms. "])
    b.add("[doctor] alright , " + title).phi(p["last"] if title else p["first"], "PERSON")
    b.tpl(" , we'll see you on {date} . take care .\n", {"date": (f"{m} {d}th", "DATE")})
    return b


def note(p: dict, rng: random.Random) -> Builder:
    b = Builder()
    C = _cap
    b.tpl("CHIEF COMPLAINT\n\n{cond}.\n\nHISTORY OF PRESENT ILLNESS\n\n", {"cond": (p["cond"].capitalize(), "TRAP")})
    b.tpl(rng.choice(["{name} is a {age}-year-old {sex} with a past medical history significant for {cond} and {cond2} who presents for follow-up. ",
                      "{title} {last} is a {age}-year-old {sex} who presents today for evaluation of {cond}. {He} also has {cond2}. "]),
          {"name": (f"{C(p['first'])} {C(p['last'])}", "PERSON"), "title": ("Mr." if p["sex"] == "male" else "Ms.", None),
           "last": (C(p["last"]), "PERSON"), "age": (str(p["age"]), "AGE"), "sex": (p["sex"], None),
           "cond": (p["cond"], "TRAP"), "cond2": (p["cond2"], "TRAP"), "He": (p["he"].capitalize(), None)})
    b.tpl("{He} reports good adherence to {drug}. ", {"He": (p["he"].capitalize(), None), "drug": (C(p["drug"]), "TRAP")})
    if rng.random() < 0.6:
        b.tpl("{His} {rel}, {fam}, accompanies {him} today. ", {"His": (p["his"].capitalize(), None), "rel": (p["relation"], None),
                                                               "fam": (C(p["fam"]), "PERSON"), "him": ("him" if p["he"] == "he" else "her", None)})
    b.add("\n\nSOCIAL HISTORY\n\n")
    b.tpl("{He} works as a {occ} at {emp}. {He} lives in {city}, {state}. ", {"He": (p["he"].capitalize(), None),
          "occ": (p["occupation"], "OCCUPATION"), "emp": (C(p["employer"]), "ORGANIZATION"), "city": (C(p["city"]), "LOCATION"),
          "state": (C(p["state"]), "LOCATION")})
    if rng.random() < 0.5:
        b.tpl("Recently traveled to {country}. ", {"country": (C(p["country"]), "LOCATION")})
    b.add("\n\nRESULTS\n\n").tpl("Hemoglobin A1c is {a1c}%.\n\n", {"a1c": (str(p["a1c"]), None)})
    b.add("ASSESSMENT AND PLAN\n\n").tpl("{cond}: Continue {drug}. Follow up with Dr. {doc} at {hosp} on {date}.\n",
                                         {"cond": (p["cond"].capitalize(), "TRAP"), "drug": (C(p["drug"]), "TRAP"), "doc": (C(p["doc"]), "PERSON"),
                                          "hosp": (C(p["hospital"]), "ORGANIZATION"), "date": (f"{C(p['visit'][0])} {p['visit'][1]}, 2024", "DATE")})
    return b


def discharge(p: dict, rng: random.Random) -> Builder:
    """Hospital discharge summary style (robustness bonus): headers with IDs, phone, address, e-mail."""
    b = Builder()
    C = _cap
    m, d, y = p["dob"]
    b.add("DISCHARGE SUMMARY\n\nPatient: ").phi(f"{C(p['last'])}, {C(p['first'])}", "PERSON").add("\nMRN: ").phi(p["mrn"], "ID")
    b.add("\nDOB: ").phi(f"{m:02d}/{d:02d}/{y}", "DOB").add("\nAdmission Date: ").phi(f"0{rng.randint(1, 9)}/1{rng.randint(0, 9)}/2023", "DATE")
    b.add("\nAttending: Dr. ").phi(f"{C(p['doc'])}", "PERSON").add("\nPhone: ").phi(f"({rng.randint(200, 989)}) 555-{rng.randint(1000, 9999)}", "CONTACT")
    b.add("\nAddress: ").phi(f"{rng.randint(10, 9999)} Oak Street", "ADDRESS").add(", ").phi(C(p["city"]), "LOCATION")
    b.add("\nEmail: ").phi(f"{p['first']}.{p['last']}@example.com", "CONTACT")
    b.tpl("\n\nHOSPITAL COURSE\n\nThe patient is a {age}-year-old {sex} admitted with exacerbation of {cond}. "
          "Treated with {drug} with good response. {His} {rel} {fam} was updated by phone.\n\nDISCHARGE MEDICATIONS\n\n{drug2} daily.\n",
          {"age": (str(p["age"]), "AGE"), "sex": (p["sex"], None), "cond": (p["cond2"], "TRAP"), "drug": (C(p["drug"]), "TRAP"),
           "His": (p["his"].capitalize(), None), "rel": (p["relation"], None), "fam": (C(p["fam"]), "PERSON"), "drug2": (C(p["drug2"]), "TRAP")})
    return b


def _sentence_case(text: str, spans: list[dict]) -> str:
    """Capitalise sentence starts, 'I', proper nouns (gold PHI except ages/dates) - offsets unchanged."""
    chars = list(text)
    for s in spans:
        if s["label"] in ("PERSON", "LOCATION", "ORGANIZATION", "HOLIDAY", "MONTH", "WEEKDAY"):
            for m in re.finditer(r"\b[a-z]", text[s["start"]:s["end"]]):
                chars[s["start"] + m.start()] = chars[s["start"] + m.start()].upper()
    out = "".join(chars)
    out = re.sub(r"\bi\b(?=[ ']|$)", "I", out)
    out = re.sub(r"(^|[.?!]\s+|\]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out, flags=re.M)
    return re.sub(r"\b(mr|mrs|ms|dr)\.", lambda m: m.group(1).capitalize() + ".", out)


def generate(n: int = 100, seed: int = 7, style: str = "aci") -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = _persona(rng)
        rec = {"encounter_id": f"SYN{i:04d}", "dataset": f"synthetic-{style}"}
        if style == "discharge":
            nb = discharge(p, rng)
            db = dialogue(p, rng)
        else:
            db, nb = dialogue(p, rng), note(p, rng)
        rec["dialogue"], rec["note"] = "".join(db.parts), "".join(nb.parts)
        if style == "cased":  # same content in conventional writing style (robustness to unseen formats)
            rec["dialogue"] = _sentence_case(rec["dialogue"], db.spans)
        rec["icd10_codes"] = [{"code": p["code"], "description": p["cond"]}, {"code": p["code2"], "description": p["cond2"]}]
        rec["_gold"] = [dict(s, field="dialogue") for s in db.spans] + [dict(s, field="note") for s in nb.spans]
        rec["_traps"] = [dict(t, field="dialogue") for t in db.traps] + [dict(t, field="note") for t in nb.traps]
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------ scoring
def score(records: list[dict], outputs: list[dict], decisions: list[list[dict]]) -> dict:
    """Entity-level privacy recall (span coverage and string survival), precision, trap preservation."""
    from collections import Counter

    tot, cov, surv, policy_kept = Counter(), Counter(), Counter(), Counter()
    fp_spans, replaced_chars, fp_chars = 0, 0, 0
    traps, traps_ok = 0, 0
    for rec, out, decs in zip(records, outputs, decisions):
        repl = {f: set() for f in ("dialogue", "note")}
        for d in decs:
            if d["action"] != "keep" and d["label"] != "LAB_VALUE":
                repl[d["field"]].update(range(d["start"], d["end"]))
        gold_chars = {f: set() for f in repl}
        kept = {f: set() for f in repl}  # detected but published by policy (exact age, calendar year)
        for d in decs:
            if d["action"] == "keep" and d.get("reason_code") == "POLICY_KEEP":
                kept[d["field"]].update(range(d["start"], d["end"]))
        for g in rec["_gold"]:
            gold_chars[g["field"]].update(range(g["start"], g["end"]))
            text = rec[g["field"]]
            alnum = [i for i in range(g["start"], g["end"]) if text[i].isalnum()]
            # raw coverage keeps its historical meaning (removed only); deliberately published entities
            # are counted separately for the policy-compliant figure
            if alnum and not all(i in repl[g["field"]] for i in alnum) and \
                    all(i in repl[g["field"]] or i in kept[g["field"]] for i in alnum):
                policy_kept[g["label"]] += 1
            tot[g["label"]] += 1
            if all(i in repl[g["field"]] for i in alnum):
                cov[g["label"]] += 1
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(g['text'])}(?![A-Za-z0-9])", out[g["field"]], re.I):
                surv[g["label"]] += 1
        for d in decs:
            if d["action"] == "keep" or d["label"] == "LAB_VALUE":
                continue
            span = set(range(d["start"], d["end"]))
            text = rec[d["field"]]
            alnum = {i for i in span if text[i].isalnum()}
            replaced_chars += len(alnum)
            fp = {i for i in alnum if i not in gold_chars[d["field"]]}
            fp_chars += len(fp)
            if alnum and len(fp) == len(alnum):
                fp_spans += 1
        for t in rec["_traps"]:
            traps += 1
            traps_ok += not any(i in repl[t["field"]] for i in range(t["start"], t["end"]))
    labels = sorted(tot)
    return {
        "entities": sum(tot.values()),
        "recall_coverage": round(sum(cov.values()) / max(1, sum(tot.values())), 4),
        # NOT unconditional privacy recall: deliberately published entities (exact age, years) count as compliant
        "policy_compliant_recall_coverage": round((sum(cov.values()) + sum(policy_kept.values())) / max(1, sum(tot.values())), 4),
        "kept_by_policy": dict(policy_kept),
        "recall_string": round(sum(surv.values()) / max(1, sum(tot.values())), 4),
        "per_label_recall_coverage": {k: round(cov[k] / tot[k], 3) for k in labels},
        "per_label_recall_string": {k: round(surv[k] / tot[k], 3) for k in labels},
        "char_precision": round(1 - fp_chars / max(1, replaced_chars), 4),
        "spurious_replacements": fp_spans,
        "trap_preservation": round(traps_ok / max(1, traps), 4),
        "traps": traps,
    }
