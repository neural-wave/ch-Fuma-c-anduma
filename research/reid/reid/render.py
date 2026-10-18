"""Render one encounter (fictional person + Synthea clinical facts) as MedAnon-schema dialogue + note.

Template families: every phrasing list is split by index parity; family 0 ("dev") uses even variants, family 1
("test") odd ones, so the blind test never sees development phrasings. Returns the text plus
  facts    - clinical elements with their surface (for clinical-utility scoring, evaluator only)
  exposure - which quasi-identifiers were actually written into the original text (evaluator only)
"""
from __future__ import annotations

import random
import re

from . import blocks as B
from .vocab import MARITAL_WORDS, WORK_PATTERNS

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
MONTHS = "January February March April May June July August September October November December".split()


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))


def year_words(y: int) -> str:
    if 2000 <= y < 2010:
        return "two thousand" + ("" if y == 2000 else " " + ONES[y - 2000])
    hi, lo = divmod(y, 100)
    return words(hi) + " " + ("hundred" if lo == 0 else ("oh " + ONES[lo]) if lo < 10 else words(lo))


def ordinal(d: int) -> str:
    special = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}
    if d < 20:
        return special.get(d, ONES[d] + "th")
    if d % 10 == 0:
        return TENS[d // 10][:-1] + "ieth"
    return TENS[d // 10] + " " + ordinal(d % 10)


class R:
    """Random helper bound to one template family."""

    def __init__(self, rng: random.Random, fam: int):
        self.rng, self.fam = rng, fam

    def pick(self, variants):
        sub = variants[self.fam::2] or variants
        return self.rng.choice(sub)

    def p(self, prob: float) -> bool:
        return self.rng.random() < prob


SMALL_TALK = [
    [("D", "how was the drive in ?"), ("P", "not bad , traffic was light today ."), ("D", "good , good .")],
    [("P", "sorry i'm a little late , parking was a nightmare ."), ("D", "no worries at all .")],
    [("D", "crazy weather we're having , huh ?"), ("P", "tell me about it , it was freezing this morning ."),
     ("D", "hopefully it warms up soon .")],
    [("P", "it's been a busy week ."), ("D", "yeah ? work or home ?"), ("P", "a bit of both honestly .")],
    [("D", "did you have a good weekend ?"), ("P", "it was quiet , mostly stayed home ."), ("D", "sometimes that's the best kind .")],
    [("P", "i almost forgot about the appointment ."), ("D", "glad you made it .")],
]
FILLERS = ["um", "uh", "you know", "like", "so", "well", "okay"]


def asr(r: R, s: str) -> str:
    """Lowercase ASR-like turn with sparse fillers and spaced punctuation."""
    s = s.lower()
    if r.p(0.25):
        s = r.rng.choice(FILLERS) + " , " + s
    s = re.sub(r"([,.?])(?=\s|$)", r" \1", s)  # keep decimals ("0.4") intact
    return re.sub(r"\s+", " ", s).strip()


def render_encounter(person: dict, enc: dict, fam: int, rng: random.Random) -> tuple[str, str, list, dict]:
    r = R(rng, fam)
    d_lines: list[str] = []
    facts: list[dict] = []
    exp: dict = {}
    sex = person["sex"]
    he, his = ("he", "his") if sex == "M" else ("she", "her")
    title = "mr." if sex == "M" else r.rng.choice(["mrs.", "ms."])
    age, child = enc["age"], enc["age"] < 18
    first, last = person["first"], person["last"]
    vdate = enc["date"]

    def D(s):
        d_lines.append("[doctor] " + asr(r, s))

    def P(s):
        d_lines.append("[patient] " + asr(r, s))

    # ---- greeting (name exposure)
    if child and r.p(0.6):
        exp["name_dialogue"] = True
        D(r.pick([f"hi {first} , and hi mom , how are we doing ?", f"hey {first} , look how big you got .",
                  f"hello {first} , hi dad , come on in .", f"hi there {first} , have a seat on the table ."]))
    elif r.p(0.6):
        exp["name_dialogue"] = True
        D(r.pick([f"hi {title} {last} , how are you doing today ?", f"good morning {first} , nice to see you .",
                  f"hello , come on in . you must be {first} {last} ?", f"hey {first} , how have you been ?",
                  f"{title} {last} ? hi , i'm the doctor seeing you today .", f"hi {first} , have a seat ."]))
    else:
        D(r.pick(["hi there , how are you doing today ?", "good morning , come on in .", "hello , nice to meet you .",
                  "hi , how are things ?"]))
    P(r.pick(["i'm okay , thanks .", "hanging in there .", "not too bad .", "i've been better honestly .",
              "good , good , thanks for seeing me .", "fine i guess ."]))
    if r.p(0.5):
        for who, txt in r.pick(SMALL_TALK):
            (D if who == "D" else P)(txt)
    # ---- identity confirmation (DOB) / age
    bd = person["birthdate"]
    if r.p(0.25):
        exp["dob"] = True
        D(r.pick(["before we start , can you confirm your date of birth for me ?", "can i get your date of birth ?",
                  "just to confirm , what's your date of birth ?", "and your birthday is ?"]))
        if r.p(0.5):
            P(f"{MONTHS[bd.month - 1]} {ordinal(bd.day)} , {year_words(bd.year)}")
        else:
            P(f"{bd.month}/{bd.day}/{str(bd.year)[2:]}")
    if r.p(0.25) and not child:
        exp["age_dialogue"] = True
        D(r.pick(["and how old are you now ?", "remind me how old you are ?", "you're how old now ?", "how old are you ?"]))
        P(r.pick([f"i'm {words(age)}", f"i just turned {words(age)}", f"{age}", f"{words(age)} , can you believe it"]))
    # ---- reason for visit
    rk, rlay = enc["reason_kind"], enc["reason_lay"]
    rel = r.rng.choice(["two days", "three days", "about a week", "a couple of weeks", "four or five days", "a month"])
    if rk == "checkup":
        D(r.pick(["so you're here for your yearly physical .", "this is your annual check up , right ?",
                  "so today is just your routine exam .", "we're doing your physical today ."]))
        P(r.pick(["yeah , just the usual .", "yep , nothing major going on .", "yes , i think i'm due .", "right ."]))
    elif rk == "chronic":
        D(r.pick([f"you're here to follow up on your {rlay} .", f"so we're checking in on the {rlay} today .",
                  f"how has the {rlay} been since last time ?", f"let's talk about your {rlay} ."]))
        P(r.pick(["it's been okay mostly .", "i think it's a little better , not sure .", "up and down honestly .",
                  "about the same ."]))
    else:
        D(r.pick(["so what brings you in today ?", "what's going on ?", "what can i do for you today ?",
                  "tell me what's been happening ."]))
        P(r.pick([f"i think i have {rlay} , it started {rel} ago .", f"i've been dealing with {rlay} for {rel} now .",
                  f"the urgent care said it might be {rlay} . it's been {rel} .", f"probably {rlay} , i'm not sure , maybe {rel} ."]))
        facts.append({"type": "duration", "surface": f"{rel} ago" if "ago" in d_lines[-1] else rel, "field": "dialogue"})
    for who, txt in B.hpi_dialogue(r, rk, rlay, child):
        (D if who == "D" else P)(txt)
    # ---- past medical history with onset years (clinical chronology + attacker-C knowledge)
    chronic = [c for c in enc["chronic"] if c["lay"] != rlay][:4]
    if chronic:
        D(r.pick(["any other medical problems i should know about ?", "what other conditions do you have ?",
                  "remind me of your medical history .", "anything else going on medically ?"]))
        bits = []
        for c in chronic:
            y = c["start"].year
            mode = r.rng.random()
            if mode < 0.45:
                yy = y if r.p(0.8) else y + r.rng.choice([-1, 1])  # remembered year, sometimes off by one
                bits.append(r.pick([f"{c['lay']} , they found that in {yy}", f"{c['lay']} since {yy} i think",
                                    f"{c['lay']} , diagnosed back in {yy}", f"{c['lay']} , that was {yy} maybe {yy + 1}"]))
                facts.append({"type": "event_year", "term": c["lay"], "year": yy, "field": "dialogue"})
                exp.setdefault("event_years", []).append(yy)
            elif mode < 0.75:
                n = max(1, vdate.year - y)
                bits.append(f"{c['lay']} for about {words(n)} years")
                facts.append({"type": "duration", "surface": f"for about {words(n)} years", "field": "dialogue"})
            else:
                bits.append(c["lay"])
            facts.append({"type": "diagnosis", "surface": c["lay"], "field": "dialogue"})
        P(" , and ".join(bits) + " .")
    # ---- surgeries and visible life events
    evs = enc["events"][:3]
    if evs and r.p(0.8):
        D(r.pick(["any surgeries or injuries in the past ?", "have you had any operations ?",
                  "any hospital stays or broken bones ?", "surgical history ?"]))
        bits = []
        for e in evs:
            if r.p(0.6):
                bits.append(r.pick([f"{e['lay']} in {e['year']}", f"{e['lay']} back in {e['year']}",
                                    f"{e['lay']} , i want to say {e['year']}", f"{e['lay']} in {year_words(e['year'])}"]))
                exp.setdefault("event_years", []).append(e["year"])
                facts.append({"type": "event_year", "term": e["lay"], "year": e["year"], "field": "dialogue"})
            else:
                n = max(1, vdate.year - e["year"])
                bits.append(f"{e['lay']} {words(n)} years ago")
        P(" , ".join(bits) + " .")
    # ---- medications
    if enc["meds"]:
        D(r.pick(["what medications are you on right now ?", "are you still taking everything on your list ?",
                  "what are you taking at the moment ?", "let's go over your meds ."]))
        P(" , ".join(f"{m['drug']} {m['dose']} milligrams" if m["dose"] else m["drug"] for m in enc["meds"][:4]) + " .")
        for m in enc["meds"][:4]:
            facts.append({"type": "medication", "surface": f"{m['drug']} {m['dose']}".strip(), "field": "dialogue"})
    if enc["meds"] and r.p(0.6):
        m = enc["meds"][0]
        D(r.pick([f"and the {m['drug']} , are you taking it every day ?", f"any problems with the {m['drug']} ?",
                  f"how are you doing on the {m['drug']} ?", f"remember to take the {m['drug']} with food ."]))
        P(r.pick(["every morning , yeah .", "no problems .", "sometimes i forget at night .", "okay , i will ."]))
    D(r.pick(["any allergies to medications ?", "allergies ?", "are you allergic to anything ?", "still no drug allergies ?"]))
    P(r.pick(["no , none", "penicillin gives me a rash", "not that i know of", "sulfa i think"]))
    # ---- social history: occupation, work pattern, family, hobby, residence, travel
    occ = person.get("occupation")
    st = person["occ_status"]
    if not child and occ and r.p(0.55):
        D(r.pick(["what do you do for work ?", "are you working these days ?", "what kind of work do you do ?",
                  "remind me what you do for a living ?"]))
        if st == "working":
            P(r.pick([f"i'm a {occ}", f"i work as a {occ}", f"still working as a {occ} , yeah", f"{occ} , going on {words(min(30, max(2, age - 22)))} years now"]))
        elif st == "retired":
            P(r.pick([f"i'm retired , i was a {occ} for a long time", f"retired now , used to be a {occ}",
                      f"i retired from being a {occ}", f"oh i'm retired , {occ} before that"]))
        else:
            P(r.pick([f"i was a {occ} but i'm out of work right now", f"between jobs , i used to work as a {occ}"]))
        exp["occupation"] = True
        if person.get("work_pattern") and r.p(0.6):
            P(WORK_PATTERNS[person["work_pattern"]][0] + " .")
            exp["work_pattern"] = True
    elif st == "student" and r.p(0.4):
        P(r.pick(["i'm a student , so school keeps me busy .", "i'm in college right now ."]))
    if not child and r.p(0.5):
        mw, partners = MARITAL_WORDS.get(person["marital"], MARITAL_WORDS["S"])
        kids = person["children"]
        exp["marital"] = True
        if person["marital"] == "M":
            partner = partners[0] if sex == "F" else partners[1]
        else:
            partner = None
        kid_txt = "" if kids is None else (f" and our {words(kids)} kids" if kids else "")
        if r.p(0.6) and kids is not None:
            exp["children"] = True
            if partner:
                P(r.pick([f"i live with {partner}{kid_txt} .", f"i'm married , we have {words(kids)} kids .",
                          f"it's me , {partner}{kid_txt} at home .", f"married , {words(kids)} children ."]))
            else:
                P(r.pick([f"i'm {mw} , {words(kids)} kids .", f"i'm {mw} . i have {words(kids)} children .",
                          f"{mw} , with {words(kids)} kids .", f"just me , i'm {mw} . {words(kids)} children ."]))
        else:
            P(r.pick([f"i'm {mw} .", f"i'm {mw} , yeah .", f"{mw} , no changes there .", f"i'm {mw} now ."]))
    if person.get("hobby") and r.p(0.3):
        P(r.pick([f"i've been doing a lot of {person['hobby']} lately .", f"{person['hobby']} is kind of my thing .",
                  f"i try to keep up with {person['hobby']} on weekends .", f"mostly {person['hobby']} to stay sane ."]))
        exp["hobby"] = True
    if r.p(0.2):
        P(r.pick([f"we moved to {person['city'].lower()} last year .", f"i live over in {person['city'].lower()} .",
                  f"it's a long drive in from {person['city'].lower()} .", f"i'm still out in {person['city'].lower()} ."]))
        exp["city"] = True
    if enc.get("travel") and r.p(0.5):
        P(r.pick([f"i just got back from {enc['travel']} .", f"we were in {enc['travel']} for two weeks ."]))
    ros_d, ros_n = B.ros(r, r.rng.randint(3, 6))
    for who, txt in ros_d:
        (D if who == "D" else P)(txt)
    ex_d, ex_n = B.exam(r, r.rng.randint(3, 6), r.rng.randint(5, 6))
    for txt in ex_d:
        D(txt)
    # ---- vitals spoken
    v = enc["vitals"]
    if v.get("weight") and r.p(0.3):
        exp["weight"] = True
        D(r.pick([f"your weight today is {round(v['weight'])} kilos .", f"you're at {round(v['weight'] * 2.2046)} pounds today .",
                  f"the scale says {round(v['weight'])} kilograms .", f"you weighed in at {round(v['weight'] * 2.2046)} pounds ."]))
    # ---- labs spoken
    for lab in enc["labs"][:2]:
        if r.p(0.5):
            D(f"your {lab['spoken']} came back at {lab['value']} .")
            facts.append({"type": "lab", "name": lab["spoken"], "value": lab["value"], "field": "dialogue"})
    # ---- plan
    pl_d, pl_n = B.plan(r, r.rng.randint(2, 4), r.rng.randint(2, 4))
    D(r.pick([f"so for the {rlay} , here's what i'm thinking .", "okay , so here's the plan .",
              f"alright , about the {rlay} .", "let's talk about next steps ."]) if rk != "checkup" else "everything looks good overall .")
    for txt in pl_d:
        D(txt)
        if r.p(0.4):
            P(r.pick(["okay .", "got it .", "makes sense .", "will do ."]))
    wk = r.rng.choice([2, 4, 6, 8, 12])
    D(r.pick([f"okay , let's see you back in {words(wk)} weeks .", f"come back in about {words(wk)} weeks and we'll recheck .",
              f"we'll follow up in {words(wk)} weeks .", f"i want to see you again in {words(wk)} weeks ."]))
    P(r.pick(["sounds good , thank you .", "okay , thanks doc .", "alright .", "thank you so much ."]))
    if r.p(0.3):
        recap = f"okay , for the note : patient is a {words(age)} year old {'male' if sex == 'M' else 'female'}"
        exp["age_dialogue"] = True
        if chronic:
            recap += f" with a history of {chronic[0]['lay']}"
        D(recap + r.pick([f" , seen today for {rlay} .", " , exam as discussed .", f" . assessment : {rlay} .", " , plan as above ."]))
    dialogue = "\n".join(d_lines)

    # ================= note (cased)
    n: list[str] = []
    if r.p(0.5):
        exp["visit_date"] = True
        n.append(r.pick([f"Date of Service: {vdate.month:02d}/{vdate.day:02d}/{vdate.year}",
                         f"Visit date: {MONTHS[vdate.month - 1]} {vdate.day}, {vdate.year}",
                         f"DATE OF VISIT: {vdate.isoformat()}", f"Encounter date: {vdate.month}/{vdate.day}/{vdate.year}"]))
    if r.p(0.3):
        exp["dob"] = True
        n.append(r.pick([f"DOB: {bd.month:02d}/{bd.day:02d}/{bd.year}", f"Date of birth: {MONTHS[bd.month - 1]} {bd.day}, {bd.year}"]))
    if r.p(0.3):
        n.append(f"MRN: {r.rng.randint(1000000, 9999999)}")
    n.append("")
    n.append(r.pick(["CHIEF COMPLAINT", "Chief Complaint:", "REASON FOR VISIT", "CC:"]))
    n.append({"checkup": "Annual physical examination.", "chronic": f"Follow-up of {rlay}."}.get(rk, f"{rlay[0].upper() + rlay[1:]}."))
    n.append("")
    n.append(r.pick(["HISTORY OF PRESENT ILLNESS", "HPI:", "History of Present Illness", "SUBJECTIVE"]))
    sexw = ("boy" if age < 13 else "young man" if age < 18 else "male") if sex == "M" else \
        ("girl" if age < 13 else "young woman" if age < 18 else "female")
    race = enc["race_word"] + " " if enc.get("race_word") and r.p(0.25) else ""
    if race:
        exp["race"] = True
    nm = f"{first} {last}" if r.p(0.5) else f"{'Mr.' if sex == 'M' else 'Ms.'} {last}"
    use_name = r.p(0.7)
    if use_name:
        exp["name_note"] = True
    subj = nm if use_name else "The patient"
    age_txt = r.pick([f"{age}-year-old", f"{age} year old", f"{age}-year-old", f"{age} y/o"]) if r.p(0.85) else None
    if age_txt:
        exp["age_note"] = True
    hx = f" with a history of {', '.join(c['lay'] for c in chronic[:3])}" if chronic else ""
    why = {"checkup": "for an annual physical", "chronic": f"for follow-up of {rlay}"}.get(rk, f"with {rlay}")
    n.append(f"{subj} is a {age_txt + ' ' if age_txt else ''}{race}{sexw}{hx} who presents {why}." if age_txt or race
             else f"{subj} presents {why}{hx.replace(' with', ', with') if hx else ''}.")
    if child:
        exp["peds_age"] = True
        facts.append({"type": "peds_age", "value": age, "field": "note"})
    n.append(r.pick([f"{he.capitalize()} reports symptoms for {rel}." if rk not in ("checkup", "chronic") else
                     f"{he.capitalize()} reports no new complaints.",
                     f"Symptoms began approximately {rel} ago." if rk not in ("checkup", "chronic") else "No acute concerns today."]))
    if chronic:
        n.append("")
        n.append(r.pick(["PAST MEDICAL HISTORY", "PMH:", "Past Medical History", "MEDICAL HISTORY"]))
        for c in chronic:
            if r.p(0.5):
                y = c["start"].year
                n.append(r.pick([f"- {c['term']} (diagnosed {y})", f"- {c['term']}, since {y}",
                                 f"- {c['term']}, dx {y}", f"- {c['term']} ({y})"]))
                facts.append({"type": "event_year", "term": c["term"], "year": y, "field": "note"})
                exp.setdefault("event_years", []).append(y)
            elif r.p(0.3):
                ds = c["start"]
                n.append(f"- {c['term']}, diagnosed {ds.month:02d}/{ds.day:02d}/{ds.year}")
                facts.append({"type": "event_date", "term": c["term"], "year": ds.year, "field": "note"})
                exp.setdefault("event_years", []).append(ds.year)
            else:
                n.append(f"- {c['term']}")
            facts.append({"type": "diagnosis", "surface": c["term"], "field": "note"})
    surg = [e for e in evs if e["kind"] == "surgery"]
    if surg:
        n.append("")
        n.append(r.pick(["PAST SURGICAL HISTORY", "PSH:", "Surgical History", "SURGICAL HISTORY"]))
        for e in surg:
            n.append(f"- {e['term']} ({e['year']})")
            facts.append({"type": "event_year", "term": e["term"], "year": e["year"], "field": "note"})
            exp.setdefault("event_years", []).append(e["year"])
    n.append("")
    n.append(r.pick(["SOCIAL HISTORY", "Social History:", "SOCIAL HX", "Social history"]))
    soc = []
    if not child and occ and r.p(0.45):
        exp["occupation"] = True
        soc.append({"working": f"Works as a {occ}.", "retired": f"Retired {occ}."}.get(st, f"Formerly worked as a {occ}; currently unemployed."))
        if person.get("work_pattern") and r.p(0.5):
            soc.append(f"{he.capitalize()} {WORK_PATTERNS[person['work_pattern']][1]}.")
            exp["work_pattern"] = True
    if not child and r.p(0.5):
        mw = MARITAL_WORDS.get(person["marital"], MARITAL_WORDS["S"])[0]
        exp["marital"] = True
        kids = person["children"]
        if kids is not None and r.p(0.5):
            exp["children"] = True
            soc.append(r.pick([f"{mw.capitalize()} with {words(kids)} children.", f"{mw.capitalize()}, {kids} children.",
                               f"{mw.capitalize()}; has {words(kids)} children.", f"{mw.capitalize()}, father/mother of {kids}."
                               .replace("father/mother", "father" if sex == "M" else "mother")]))
        else:
            soc.append(f"{mw.capitalize()}.")
    if child:
        soc.append(f"Lives with {his} parents. Attends school." if age >= 5 else "Lives with parents.")
    if person.get("hobby") and r.p(0.2):
        exp["hobby"] = True
        soc.append(f"Enjoys {person['hobby']}.")
    if r.p(0.15):
        exp["city"] = True
        soc.append(f"Lives in {person['city']}.")
    soc.append(r.pick(["Denies tobacco use.", "Former smoker.", "Drinks alcohol socially.", "No illicit drug use."]))
    n.append(" ".join(soc))
    n.append("")
    n.append(r.pick(["FAMILY HISTORY", "Family History:", "FAMILY HX", "Family history"]))
    n.append(" ".join(r.rng.sample(B.FAMILY[r.fam::2], 2)))
    if enc["meds"]:
        n.append("")
        n.append(r.pick(["MEDICATIONS", "Current Medications:", "MEDS", "Medications"]))
        for m in enc["meds"][:6]:
            n.append(f"- {m['drug'].capitalize()} {m['dose']} mg" if m["dose"] else f"- {m['drug'].capitalize()}")
            facts.append({"type": "medication", "surface": f"{m['drug'].capitalize()} {m['dose']}".strip(), "field": "note"})
    n.append("")
    n.append(r.pick(["ALLERGIES", "Allergies:", "ALLERGIES", "Allergies"]))
    n.append(r.rng.choice(B.ALLERGY[r.fam::2]))
    n.append("")
    n.append(r.pick(["REVIEW OF SYSTEMS", "ROS:", "Review of Systems", "ROS"]))
    n += ros_n
    n.append("")
    n.append(r.pick(["VITALS", "Vital Signs:", "VITAL SIGNS", "Vitals:"]))
    vs = []
    if v.get("sbp"):
        vs.append(f"BP {v['sbp']}/{v['dbp']}")
        facts.append({"type": "vital", "surface": f"{v['sbp']}/{v['dbp']}", "field": "note"})
    if v.get("hr"):
        vs.append(f"HR {v['hr']}")
    us = r.p(0.5)
    needed = child or enc.get("weight_dosing", False)
    if v.get("height") and r.p(0.6):
        exp["height"] = True
        if us and not child:
            inch = round(v["height"] / 2.54)
            vs.append(f"Height {inch // 12}'{inch % 12}\"")
        else:
            vs.append(f"Height {v['height']:.0f} cm")
            facts.append({"type": "anthro", "kind": "height", "surface": f"{v['height']:.0f} cm", "needed": needed, "field": "note"})
    if v.get("weight") and r.p(0.75):
        exp["weight"] = True
        if us and not child:
            vs.append(f"Weight {round(v['weight'] * 2.2046)} lbs")
        else:
            wt = f"{v['weight']:.1f} kg"
            vs.append(f"Weight {wt}")
            facts.append({"type": "anthro", "kind": "weight", "surface": wt, "needed": needed, "field": "note"})
    if v.get("bmi") and r.p(0.5) and not child:
        exp["bmi"] = True
        vs.append(f"BMI {v['bmi']:.1f}")
        facts.append({"type": "anthro", "kind": "bmi", "surface": f"{v['bmi']:.1f}", "needed": False, "field": "note"})
    n.append(", ".join(vs) + "." if vs else "Within normal limits.")
    n.append("")
    n.append(r.pick(["PHYSICAL EXAMINATION", "Physical Exam:", "PHYSICAL EXAM", "Exam:"]))
    n += ex_n
    if enc["labs"]:
        n.append("")
        n.append(r.pick(["RESULTS", "Labs:", "LABORATORY DATA", "Recent labs:"]))
        n.append("; ".join(f"{lab['name']} {lab['value']} {lab['unit']}".strip() for lab in enc["labs"]) + ".")
        for lab in enc["labs"]:
            facts.append({"type": "lab", "name": lab["name"], "value": lab["value"], "field": "note"})
    n.append("")
    n.append(r.pick(["ASSESSMENT AND PLAN", "Assessment/Plan:", "ASSESSMENT & PLAN", "IMPRESSION AND PLAN"]))
    pdx = enc["reason_term"]
    n.append(f"1. {pdx}. " + r.pick(["Supportive care discussed.", "Continue current management.",
                                     "Plan reviewed with the patient.", "Return precautions given."]))
    facts.append({"type": "diagnosis", "surface": pdx, "field": "note"})
    for i, c in enumerate(chronic, 2):
        n.append(f"{i}. {c['term']}. " + r.pick(["Stable.", "Continue current regimen.", "Monitor.", "Reassess at follow-up."]))
    n += pl_n
    if enc.get("weight_dosing") and v.get("weight"):
        n.append(f"- Weight-based dosing: {enc['weight_dosing']} calculated for {v['weight']:.1f} kg.")
        facts.append({"type": "anthro", "kind": "weight", "surface": f"{v['weight']:.1f} kg", "needed": True, "field": "note"})
    n.append(f"Follow-up in {wk} weeks.")
    facts.append({"type": "duration", "surface": f"{wk} weeks", "field": "note"})
    return dialogue, "\n".join(n), facts, exp
