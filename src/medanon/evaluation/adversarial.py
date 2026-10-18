"""Adversarial synthetic evaluation set (frozen; independent of the tuning templates in synthetic.py).

Every encounter is assembled from *disjoint* lexicons and phrasings (no template from synthetic.py
or canary.py is reused) and carries exact gold spans for:
  * PHI (label + `hard` tag describing the difficulty, e.g. "oov-name", "drug-name-collision"),
  * clinical TRAPS that must survive byte-for-byte, typed by what they protect
    ("negation", "family-history", "dosage", "lab-value", "eponym", "temporal", "medication").
The set is used only for final comparison, never for rule development.
"""
from __future__ import annotations

import random

# lexicons deliberately disjoint from synthetic.py / canary.py
OOV_FIRST = ["Zoë", "Søren", "Ngozi", "Xóchitl", "Iñaki", "Bartłomiej", "Oluwadamilare", "Aigerim", "Thảo", "Eilidh",
             "Radhika", "Kwabena", "Ágnes", "Dagný", "Teodora", "Mihai", "Yusuf", "Saoirse", "Ikaika", "Tupac"]
OOV_LAST = ["Nwachukwu", "Szymańska", "Ó Briain", "Van der Merwe", "Abu-Haidar", "Gonçalves", "Rautenbach",
            "Wójcik", "Nakamura-Liu", "Østergaard", "Bianchi-Ferraro", "Tshabalala", "Kowalczyk", "Al-Rashidi"]
COMMON_WORD_NAMES = ["Hope", "Will", "Grace", "Rich", "Joy", "Faith", "Summer", "Sunny", "Hunter", "Chance"]
DRUG_NAMES = ["Allegra", "Lyrica", "Flomax", "Valium", "Keflex", "Paxil"]   # also used as first names
EPONYM_SURNAMES = ["Hodgkin", "Cushing", "Raynaud", "Barrett", "Paget", "Huntington"]
ORDINARY_ORGS = ["Apple", "Target", "Delta", "Shell", "Chase", "Subway"]
CITIES = ["Tromsø", "Kraków", "Guadalajara", "Lagos", "Da Nang", "Reykjavík", "Bozeman", "Yakima", "Natchez", "Kenosha"]
JOBS = [("crane operator", "manual labor"), ("dental hygienist", "healthcare"), ("air traffic controller", "transport"),
        ("sommelier", "service"), ("court stenographer", "office"), ("beekeeper", "agriculture")]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class _B:
    def __init__(self):
        self.text, self.gold, self.traps = "", [], []

    def t(self, s: str):
        self.text += s
        return self

    def phi(self, s: str, label: str, hard: str):
        self.gold.append({"start": len(self.text), "end": len(self.text) + len(s), "label": label, "hard": hard, "text": s})
        self.text += s
        return self

    def trap(self, s: str, kind: str):
        self.traps.append({"start": len(self.text), "end": len(self.text) + len(s), "kind": kind, "text": s})
        self.text += s
        return self


def _spoken(n: int, rng: random.Random) -> str:
    return " ".join(rng.choice(DIGITS) for _ in range(n))


def encounter(i: int, rng: random.Random) -> dict:
    first, last = rng.choice(OOV_FIRST), rng.choice(OOV_LAST)
    fl, ll = first.lower(), last.lower()
    relative = rng.choice(COMMON_WORD_NAMES)
    drugname_person = rng.choice(DRUG_NAMES)
    epo = rng.choice(EPONYM_SURNAMES)
    org = rng.choice(ORDINARY_ORGS)
    city = rng.choice(CITIES)
    job, _ = rng.choice(JOBS)
    age = rng.randint(23, 88)
    d, n = _B(), _B()
    # ---------------- dialogue (lowercase ASR style)
    d.t("[doctor] okay , so before we start , can you spell your last name for me ?\n[patient] sure , it's ")
    d.phi(ll, "PERSON", "oov-name").t(" .\n")
    d.t("[doctor] and your first name is ").phi(fl, "PERSON", "oov-name").t(" , right ?\n[patient] yes .\n")
    d.t("[doctor] what's the best number to reach you ?\n[patient] it's ").phi(_spoken(10, rng), "CONTACT", "spoken-digits").t(" .\n")
    d.t("[doctor] and your chart number ?\n[patient] i think it's ").phi(_spoken(7, rng), "ID", "spoken-digits").t(" .\n")
    d.t("[patient] ").phi(relative.lower(), "PERSON", "common-word-name").t(" usually drives me , she's my niece .\n")
    d.t("[patient] my friend ").phi(drugname_person.lower(), "PERSON", "drug-name-collision").t(" recommended this clinic .\n")
    d.t("[doctor] are you still taking the ").trap("lyrica", "medication").t(" ?\n[patient] yes , ").trap("seventy five milligrams twice a day", "dosage").t(" .\n")
    d.t("[doctor] any ").trap("chest pain", "negation").t(" ?\n[patient] ").trap("no chest pain", "negation").t(" , and ").trap("i don't have any shortness of breath", "negation").t(" .\n")
    d.t("[patient] my ").trap("father had colon cancer at fifty", "family-history").t(" .\n")
    d.t("[patient] i work as a ").phi(job, "OCCUPATION", "rare-occupation").t(" at ").phi(org.lower(), "ORGANIZATION", "ordinary-word-org").t(" .\n")
    d.t("[patient] we moved from ").phi(city.lower(), "LOCATION", "oov-city").t(" last spring .\n")
    # a very long turn with the identifier at its end
    filler = "um , and then the pain kind of comes and goes , mostly in the evening after i walk the dog , " * 18
    d.t("[patient] " + filler + "anyway my neighbor ").phi(rng.choice(OOV_FIRST).lower(), "PERSON", "end-of-long-turn").t(" checks on me .\n")
    d.t("[doctor] the ").trap(f"{epo.lower()} lymphoma" if epo == "Hodgkin" else f"{epo.lower()}'s disease" if epo != "Barrett" else "barrett's esophagus", "eponym")
    d.t(" history is noted . and doctor ").phi(epo.lower(), "PERSON", "eponym-surname").t(" will follow up .\n")
    d.t("[doctor] your ").trap("a1c is eight point one", "lab-value").t(" , we'll recheck it ").trap("in three months", "temporal").t(" .\n")
    # ---------------- note (cased)
    n.t("HISTORY OF PRESENT ILLNESS\n\n").phi(f"{first} {last}", "PERSON", "oov-name").t(" is a ").phi(str(age), "AGE", "age")
    n.t("-year-old patient who presents with knee pain. ").trap("Denies fever or chills", "negation").t(". ")
    n.t("Lives in ").phi(city, "LOCATION", "oov-city").t(" with niece ").phi(relative, "PERSON", "common-word-name").t(". ")
    n.t("Works as a ").phi(job, "OCCUPATION", "rare-occupation").t(" at ").phi(org, "ORGANIZATION", "ordinary-word-org").t(".\n\n")
    n.t("FAMILY HISTORY\n\n").trap("Father with colon cancer at age 50", "family-history").t(".\n\n")
    n.t("MEDICATIONS\n\n").trap("Lyrica 75 mg twice daily", "dosage").t(".\n\n")
    n.t("RESULTS\n\n").trap("Hemoglobin A1c 8.1%", "lab-value").t(".\n\n")
    n.t("ASSESSMENT AND PLAN\n\n").trap("Knee osteoarthritis", "diagnosis").t(". Follow up with Dr. ").phi(epo, "PERSON", "eponym-surname")
    n.t(". Recheck A1c ").trap("in 3 months", "temporal").t(".\n")
    return {"encounter_id": f"ADV{i:03d}", "dataset": "adversarial", "dialogue": d.text, "note": n.text,
            "icd10_codes": [{"code": "M17.11", "description": "Unilateral primary osteoarthritis, right knee"}],
            "_gold": [dict(g, field="dialogue") for g in d.gold] + [dict(g, field="note") for g in n.gold],
            "_traps": [dict(t, field="dialogue") for t in d.traps] + [dict(t, field="note") for t in n.traps]}


# ------------------------------------------------------------------------------- held-out TEST bank
# Disjoint lexicons and different phrasings; never inspected while developing rules.
T_FIRST = ["Anaïs", "Björn", "Chidi", "Dorotea", "Émile", "Farrukh", "Göran", "Hyun-woo", "Ilse", "Joaquín", "Kalani",
           "Lúcia", "Mbali", "Nuno", "Orla", "Pilar", "Quang", "Rasmus", "Sione", "Tadeusz"]
T_LAST = ["Lindgren", "Achterberg", "Okonjo", "Fitzwilliam", "Dąbrowski", "Moreau-Duval", "Halvorsen", "Ikeda",
          "Petrović", "Sandoval-Reyes", "Quispe", "Mwangi", "Oyelaran", "Brannigan"]
T_COMMON_NAMES = ["Rose", "Mark", "Bill", "Dawn", "June", "Frank", "Penny", "Sandy", "Gene", "Ray"]
T_DRUG_PERSONS = ["Celexa", "Ambien", "Zyrtec", "Ativan"]
T_EPONYM_SURNAMES = ["Wilson", "Addison", "Graves", "Bell", "Turner"]
T_ORGS = ["Amazon", "Uber", "Costco", "Sprint", "Visa", "Lowe's"]
T_CITIES = ["Ljubljana", "Hilo", "Tucumcari", "Poughkeepsie", "Dushanbe", "Kaunas", "Eau Claire", "Puyallup"]
T_JOBS = ["forklift mechanic", "museum curator", "hospice chaplain", "wind turbine technician", "tattoo artist",
          "lighthouse keeper", "radio dispatcher", "glassblower"]


def encounter_test(i: int, rng: random.Random) -> dict:
    first, last = rng.choice(T_FIRST), rng.choice(T_LAST)
    sib = rng.choice(T_COMMON_NAMES)
    drug_p = rng.choice(T_DRUG_PERSONS)
    epo = rng.choice(T_EPONYM_SURNAMES)
    org, city, job = rng.choice(T_ORGS), rng.choice(T_CITIES), rng.choice(T_JOBS)
    age = rng.randint(30, 85)
    d, n = _B(), _B()
    d.t("[doctor] good afternoon . remind me how you'd like to be addressed ?\n[patient] just ").phi(first.lower(), "PERSON", "oov-name").t(" is fine .\n")
    d.t("[doctor] and the surname on the chart is ").phi(last.lower(), "PERSON", "oov-name").t(" ?\n[patient] correct .\n")
    d.t("[doctor] okay , confirming your callback number , ").phi(_spoken(10, rng), "CONTACT", "spoken-digits").t(" ?\n[patient] yep .\n")
    d.t("[doctor] and member id ").phi(_spoken(8, rng), "ID", "spoken-digits").t(" for the insurance .\n")
    d.t("[patient] my sister ").phi(sib.lower(), "PERSON", "common-word-name").t(" is coming to the next visit .\n")
    d.t("[patient] honestly ").phi(drug_p.lower(), "PERSON", "drug-name-collision").t(" , my coworker , told me to come in .\n")
    d.t("[doctor] are you taking the ").trap("ativan", "medication").t(" only at night ?\n[patient] ").trap("half a milligram at bedtime", "dosage").t(" , yeah .\n")
    d.t("[doctor] ").trap("no numbness or tingling", "negation").t(" ?\n[patient] ").trap("none at all", "negation").t(" .\n")
    d.t("[patient] ").trap("my mom had a stroke when she was sixty", "family-history").t(" .\n")
    d.t("[patient] i've been a ").phi(job, "OCCUPATION", "rare-occupation").t(" for years , i'm at ").phi(org.lower(), "ORGANIZATION", "ordinary-word-org").t(" now .\n")
    d.t("[patient] we're up in ").phi(city.lower(), "LOCATION", "oov-city").t(" so it's a drive .\n")
    filler = "and i guess the other thing is that the swelling gets worse when i stand at work for a long time , " * 16
    d.t("[patient] " + filler + "oh and say hi to ").phi(rng.choice(T_FIRST).lower(), "PERSON", "end-of-long-turn").t(" from me .\n")
    d.t("[doctor] with the ").trap(f"{epo.lower()}'s disease" if epo in ("Wilson", "Addison", "Graves") else ("bell's palsy" if epo == "Bell" else "turner syndrome"), "eponym")
    d.t(" in mind , doctor ").phi(epo.lower(), "PERSON", "eponym-surname").t(" from endocrinology will see you .\n")
    d.t("[doctor] the ").trap("ldl was one forty two", "lab-value").t(" so we'll recheck ").trap("in six weeks", "temporal").t(" .\n")
    n.t("HPI: ").phi(f"{last}, {first}", "PERSON", "oov-name").t(" (").phi(str(age), "AGE", "age").t(" y/o) seen for leg swelling. ")
    n.t(f"{'He' if rng.random() < .5 else 'She'} is a ").phi(job, "OCCUPATION", "rare-occupation").t(" employed by ").phi(org, "ORGANIZATION", "ordinary-word-org").t(". ")
    n.t("Resides in ").phi(city, "LOCATION", "oov-city").t("; sister ").phi(sib, "PERSON", "common-word-name").t(" assists at home. ")
    n.t(f"{'Reports' if rng.random() < .5 else 'Endorses'} ").trap("no numbness or tingling", "negation").t(".\n")
    n.t("FHx: ").trap("mother with stroke at 60", "family-history").t(".\n")
    n.t("Meds: ").trap("Ativan 0.5 mg at bedtime", "dosage").t(".\n")
    n.t("Labs: ").trap("LDL 142", "lab-value").t(".\n")
    n.t("Plan: ").trap("Chronic venous insufficiency", "diagnosis").t("; endocrinology f/u with Dr. ").phi(epo, "PERSON", "eponym-surname").t(" ").trap("in 6 weeks", "temporal").t(".\n")
    return {"encounter_id": f"ADT{i:03d}", "dataset": "adversarial-test", "dialogue": d.text, "note": n.text,
            "icd10_codes": [{"code": "I87.2", "description": "Venous insufficiency (chronic) (peripheral)"}],
            "_gold": [dict(g, field="dialogue") for g in d.gold] + [dict(g, field="note") for g in n.gold],
            "_traps": [dict(t, field="dialogue") for t in d.traps] + [dict(t, field="note") for t in n.traps]}


def generate(n: int = 40, seed: int = 2026, bank: str = "dev") -> list[dict]:
    """bank='dev' was used to discover failure classes; bank='test' is held out for final numbers."""
    rng = random.Random(seed if bank == "dev" else seed + 1000)
    build = encounter if bank == "dev" else encounter_test
    return [build(i, rng) for i in range(n)]
