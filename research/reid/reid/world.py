"""Generator (evaluator side): Synthea CSV population -> fictional persons -> encounters, attacker registers, truth.

    python -m reid.world --synthea <synthea csv dir> --out <world dir> --seed 1 --family dev [--n-encounters 300]

Layout written (only attacker/ and, after MedAnon, anonymized/ may be shown to the attacker):
  private/        persons.csv.gz (truth for the whole population), encounters_truth.jsonl, keys_<T>.csv
                  (opaque register key -> person), manifest.json
  release_input/  encounters.jsonl  (identifiable MedAnon input; never given to the attacker)
  attacker/       aux_A.parquet, aux_B.parquet, aux_C.parquet (registers of the WHOLE population; opaque keys)
Every person gets a register row through the same noisy process, target or not, so presence in a register is no
shortcut. Encounter ids and register keys are independent random draws; files are shuffled.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import random
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .render import render_encounter
from .vocab import HOBBIES, OCCUPATIONS, TRAVEL, WORK_PATTERNS

REF = date(2026, 10, 1)
WINDOW = ("2023-01-01", "2026-09-30")
# SNOMED description -> ICD-10, kind, lay phrase (dialogue), clinical term (note)
_COND = """Essential hypertension (disorder)|I10|chronic|high blood pressure|Essential hypertension
Hyperlipidemia (disorder)|E78.5|chronic|high cholesterol|Hyperlipidemia
Hypertriglyceridemia (disorder)|E78.1|chronic|high triglycerides|Hypertriglyceridemia
Diabetes mellitus type 2 (disorder)|E11.9|chronic|type 2 diabetes|Type 2 diabetes mellitus
Prediabetes (finding)|R73.03|chronic|prediabetes|Prediabetes
Metabolic syndrome X (disorder)|E88.81|chronic|metabolic syndrome|Metabolic syndrome
Ischemic heart disease (disorder)|I25.10|chronic|heart disease|Coronary artery disease
Chronic kidney disease stage 1 (disorder)|N18.1|chronic|kidney disease|Chronic kidney disease, stage 1
Chronic kidney disease stage 2 (disorder)|N18.2|chronic|kidney disease|Chronic kidney disease, stage 2
Chronic kidney disease stage 3 (disorder)|N18.30|chronic|kidney disease|Chronic kidney disease, stage 3
Chronic kidney disease stage 4 (disorder)|N18.4|chronic|kidney disease|Chronic kidney disease, stage 4
End-stage renal disease (disorder)|N18.6|chronic|kidney failure|End-stage renal disease
Disorder of kidney due to diabetes mellitus (disorder)|E11.22|chronic|diabetic kidney disease|Diabetic nephropathy
Neuropathy due to type 2 diabetes mellitus (disorder)|E11.40|chronic|nerve damage from diabetes|Diabetic neuropathy
Nonproliferative diabetic retinopathy due to type II diabetes mellitus|E11.329|chronic|diabetic eye disease|Nonproliferative diabetic retinopathy
Chronic sinusitis (disorder)|J32.9|chronic|chronic sinus problems|Chronic sinusitis
Osteoarthritis of knee (disorder)|M17.9|chronic|arthritis in my knee|Osteoarthritis of the knee
Osteoarthritis of hip (disorder)|M16.9|chronic|arthritis in my hip|Osteoarthritis of the hip
Localized  primary osteoarthritis of the hand (disorder)|M19.049|chronic|arthritis in my hands|Osteoarthritis of the hand
Osteoporosis (disorder)|M81.0|chronic|osteoporosis|Osteoporosis
Obstructive sleep apnea syndrome (disorder)|G47.33|chronic|sleep apnea|Obstructive sleep apnea
Sleep apnea (disorder)|G47.30|chronic|sleep apnea|Sleep apnea
Chronic intractable migraine without aura (disorder)|G43.019|chronic|migraines|Chronic migraine without aura
Fibromyalgia (disorder)|M79.7|chronic|fibromyalgia|Fibromyalgia
Seizure disorder (disorder)|G40.909|chronic|seizures|Seizure disorder
Epilepsy (disorder)|G40.909|chronic|epilepsy|Epilepsy
Childhood asthma (disorder)|J45.909|chronic|asthma|Asthma
Asthma (disorder)|J45.909|chronic|asthma|Asthma
Atopic dermatitis (disorder)|L20.9|chronic|eczema|Atopic dermatitis
Perennial allergic rhinitis (disorder)|J30.89|chronic|allergies|Perennial allergic rhinitis
Perennial allergic rhinitis with seasonal variation (disorder)|J30.2|chronic|seasonal allergies|Allergic rhinitis
Chronic congestive heart failure (disorder)|I50.9|chronic|heart failure|Congestive heart failure
Malignant neoplasm of breast (disorder)|C50.919|chronic|breast cancer|Malignant neoplasm of breast
Malignant neoplasm of colon (disorder)|C18.9|chronic|colon cancer|Malignant neoplasm of colon
Alzheimer's disease (disorder)|G30.9|chronic|memory problems|Alzheimer disease
Anemia (disorder)|D64.9|chronic|anemia|Anemia
Dependent drug abuse (disorder)|F19.20|chronic|substance use problems|Substance dependence
Human immunodeficiency virus infection (disorder)|B20|chronic|hiv|HIV infection
Child attention deficit disorder (disorder)|F90.9|chronic|adhd|Attention deficit hyperactivity disorder
Chronic pain (finding)|G89.29|chronic|chronic pain|Chronic pain
Chronic low back pain (finding)|M54.50|chronic|back pain|Chronic low back pain
Chronic neck pain (finding)|M54.2|chronic|neck pain|Chronic neck pain
Adolescent idiopathic scoliosis (disorder)|M41.129|chronic|scoliosis|Adolescent idiopathic scoliosis
Recurrent urinary tract infection (disorder)|N39.0|chronic|recurrent bladder infections|Recurrent urinary tract infection
Polyp of colon (disorder)|K63.5|chronic|colon polyps|Colon polyp
Viral sinusitis (disorder)|J01.90|acute|a sinus infection|Acute viral sinusitis
Acute viral pharyngitis (disorder)|J02.9|acute|a sore throat|Acute viral pharyngitis
Acute bronchitis (disorder)|J20.9|acute|bronchitis|Acute bronchitis
Otitis media (disorder)|H66.90|acute|an ear infection|Otitis media
Streptococcal sore throat (disorder)|J02.0|acute|strep throat|Streptococcal pharyngitis
Acute infective cystitis (disorder)|N30.00|acute|a bladder infection|Acute cystitis
Acute bacterial sinusitis (disorder)|J01.90|acute|a sinus infection|Acute bacterial sinusitis
Sinusitis (disorder)|J01.90|acute|a sinus infection|Sinusitis
Disease caused by severe acute respiratory syndrome coronavirus 2 (disorder)|U07.1|acute|covid|COVID-19
Pneumonia (disorder)|J18.9|acute|pneumonia|Pneumonia
Sprain of ankle (disorder)|S93.409A|injury|a sprained ankle|Sprain of ankle
Sprain of wrist (disorder)|S63.509A|injury|a sprained wrist|Sprain of wrist
Fracture of bone (disorder)|T14.8XXA|injury|a broken bone|Fracture
Fracture of forearm (disorder)|S52.90XA|injury|a broken arm|Fracture of forearm
Fracture of clavicle (disorder)|S42.009A|injury|a broken collarbone|Fracture of clavicle
Fracture of ankle (disorder)|S82.899A|injury|a broken ankle|Fracture of ankle
Fracture subluxation of wrist (disorder)|S62.109A|injury|a broken wrist|Fracture of wrist
Osteoporotic fracture of bone (disorder)|M80.00XA|injury|a fracture from weak bones|Osteoporotic fracture
Concussion injury of brain (disorder)|S06.0X0A|injury|a concussion|Concussion
Concussion with no loss of consciousness (disorder)|S06.0X0A|injury|a concussion|Concussion without loss of consciousness
Concussion with loss of consciousness (disorder)|S06.0X1A|injury|a concussion|Concussion with loss of consciousness
Whiplash injury to neck (disorder)|S13.4XXA|injury|whiplash|Whiplash injury
Injury of knee (disorder)|S89.90XA|injury|a knee injury|Injury of knee
Laceration of hand (disorder)|S61.419A|injury|a cut on my hand|Laceration of hand
Laceration of forearm (disorder)|S51.819A|injury|a cut on my arm|Laceration of forearm
Laceration of foot (disorder)|S91.319A|injury|a cut on my foot|Laceration of foot
Facial laceration (disorder)|S01.81XA|injury|a cut on my face|Facial laceration
Burn injury (morphologic abnormality)|T30.0|injury|a burn|Burn injury
Acute non-ST segment elevation myocardial infarction (disorder)|I21.4|injury|a heart attack|NSTEMI
Acute ST segment elevation myocardial infarction (disorder)|I21.3|injury|a heart attack|STEMI
Myocardial infarction (disorder)|I21.9|injury|a heart attack|Myocardial infarction
Overdose (disorder)|T50.901A|acute|an overdose|Drug overdose
Normal pregnancy (finding)|Z34.90|preg|pregnancy|Normal pregnancy, prenatal care
Pre-eclampsia (disorder)|O14.90|acute|high blood pressure in pregnancy|Pre-eclampsia
Miscarriage in first trimester (disorder)|O03.9|acute|a miscarriage|Miscarriage in first trimester"""
COND = {d: {"icd": i, "kind": k, "lay": lay, "term": t} for d, i, k, lay, t in (x.split("|") for x in _COND.splitlines())}
VISIBLE_INJURY = ("Fracture", "Concussion", "infarction")  # life events an acquaintance plausibly knows
_SURG = re.compile(r"ectomy|arthroplasty|bypass|transplant|cesarean|caesarean|replacement of|mastectomy", re.I)
_NOT_SURG = re.compile(r"tooth|dental|contracept|teeth|ligation|biopsy|polypectomy", re.I)
LABS = {  # Synthea LOINC description prefix -> (note name, spoken, unit, decimals)
    "Hemoglobin A1c/Hemoglobin.total in Blood": ("HbA1c", "a1c", "%", 1),
    "Glucose [Mass/volume] in Blood": ("Glucose", "blood sugar", "mg/dL", 0),
    "Glucose [Mass/volume] in Serum or Plasma": ("Glucose", "blood sugar", "mg/dL", 0),
    "Creatinine [Mass/volume] in Blood": ("Creatinine", "creatinine", "mg/dL", 2),
    "Creatinine [Mass/volume] in Serum or Plasma": ("Creatinine", "creatinine", "mg/dL", 2),
    "Cholesterol [Mass/volume] in Serum or Plasma": ("Total cholesterol", "cholesterol", "mg/dL", 0),
    "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay": ("LDL", "ldl", "mg/dL", 0),
    "Triglyceride [Mass/volume] in Serum or Plasma": ("Triglycerides", "triglycerides", "mg/dL", 0),
    "Potassium [Moles/volume] in Blood": ("Potassium", "potassium", "mmol/L", 1),
    "Hemoglobin [Mass/volume] in Blood": ("Hemoglobin", "hemoglobin", "g/dL", 1),
}
VITALS = {"Body Height": "height", "Body Weight": "weight", "Body mass index (BMI) [Ratio]": "bmi",
          "Systolic Blood Pressure": "sbp", "Diastolic Blood Pressure": "dbp", "Heart rate": "hr"}
CHECKUP = re.compile(r"check up|General examination|Well child", re.I)
RACE_WORD = {"white": "White", "black": "Black", "asian": "Asian", "hispanic": "Hispanic"}


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def _strip(name) -> str:
    return re.sub(r"\d+", "", str(name)).strip() if isinstance(name, str) else ""


def _med(desc: str) -> dict | None:
    if desc.startswith("{") or "Pack" in desc:
        return None
    m = re.match(r"^(?:[\d.]+ (?:HR|ML|ACTUAT) )*(?:NDA\d+ )?(?:[\d.]+ ACTUAT )?([A-Za-z][A-Za-z\- ]+?) ([\d.]+) MG\b", desc)
    if m:
        drug = re.sub(r"\b(hydrochloride|sodium|succinate|potassium|bitartrate|propionate|phosphate|acid)\b", "", m.group(1), flags=re.I)
        return {"drug": " ".join(drug.lower().split()), "dose": m.group(2)}
    b = re.search(r"\[([A-Za-z]+)\]", desc)
    return {"drug": (b.group(1) if b else " ".join(desc.split()[:2])).lower(), "dose": ""}


# ------------------------------------------------------------------ population (cached per Synthea run)
def load_population(csv: Path, cache: Path | None = None) -> dict:
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    pat = pd.read_csv(csv / "patients.csv", usecols=["Id", "BIRTHDATE", "FIRST", "LAST", "MARITAL", "RACE", "ETHNICITY",
                                                     "GENDER", "CITY", "COUNTY", "INCOME"])
    cond = pd.read_csv(csv / "conditions.csv", usecols=["START", "STOP", "PATIENT", "ENCOUNTER", "DESCRIPTION"])
    proc = pd.read_csv(csv / "procedures.csv", usecols=["START", "PATIENT", "DESCRIPTION"])
    proc = proc[proc.DESCRIPTION.str.contains(_SURG) & ~proc.DESCRIPTION.str.contains(_NOT_SURG)]
    enc = pd.read_csv(csv / "encounters.csv", usecols=["Id", "START", "PATIENT", "ENCOUNTERCLASS", "DESCRIPTION", "REASONDESCRIPTION"])
    enc = enc[(enc.START >= WINDOW[0]) & (enc.START <= WINDOW[1] + "T23:59")
              & enc.ENCOUNTERCLASS.isin(["ambulatory", "outpatient", "wellness", "urgentcare", "emergency"])
              & ~enc.DESCRIPTION.str.contains("dental|vaccine|Screening surveillance", case=False, regex=True)]
    persons = {}
    sdoh = cond[cond.DESCRIPTION.str.contains("employment|Unemployed|labor force|education|high school", case=False, regex=True)]
    sd = {}
    for r in sdoh.sort_values("START").itertuples():
        k = "edu" if "educ" in r.DESCRIPTION.lower() or "school" in r.DESCRIPTION.lower() else "emp"
        sd.setdefault(r.PATIENT, {})[k] = r.DESCRIPTION
    cmap = {}
    for r in cond[cond.DESCRIPTION.isin(COND)].itertuples():
        cmap.setdefault(r.PATIENT, []).append({"desc": r.DESCRIPTION, "start": _d(r.START),
                                               "stop": _d(r.STOP) if isinstance(r.STOP, str) else None, "enc": r.ENCOUNTER})
    pmap = {}
    for r in proc.itertuples():
        t = re.sub(r"\s*\((procedure|regime/therapy)\)", "", r.DESCRIPTION).strip()
        pmap.setdefault(r.PATIENT, {}).setdefault(t, _d(r.START))  # first occurrence of each surgery
    for r in pat.itertuples():
        bd = _d(r.BIRTHDATE)
        persons[r.Id] = {"pid": r.Id, "first": _strip(r.FIRST), "last": _strip(r.LAST), "sex": r.GENDER, "birthdate": bd,
                         "race": "hispanic" if r.ETHNICITY == "hispanic" else r.RACE,
                         "marital": r.MARITAL if isinstance(r.MARITAL, str) else None, "city": r.CITY, "county": r.COUNTY,
                         "income": float(r.INCOME) if r.INCOME == r.INCOME else 50000.0, "sdoh": sd.get(r.Id, {}),
                         "conds": cmap.get(r.Id, []), "surgeries": pmap.get(r.Id, {})}
    # latest height/weight/BMI for everybody; per-encounter vitals and labs
    need = set(VITALS) | set(LABS)
    obs = []
    for ch in pd.read_csv(csv / "observations.csv", usecols=["DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION", "VALUE"],
                          chunksize=2_000_000):
        obs.append(ch[ch.DESCRIPTION.isin(need)])
    obs = pd.concat(obs)
    obs["VALUE"] = pd.to_numeric(obs.VALUE, errors="coerce")
    latest = obs[obs.DESCRIPTION.isin(["Body Height", "Body Weight", "Body mass index (BMI) [Ratio]"])].sort_values("DATE")
    for r in latest.groupby(["PATIENT", "DESCRIPTION"]).tail(1).itertuples():
        if r.PATIENT in persons:
            persons[r.PATIENT][VITALS[r.DESCRIPTION]] = float(r.VALUE)
    eobs = {}
    for r in obs[obs.ENCOUNTER.isin(set(enc.Id))].itertuples():
        eobs.setdefault(r.ENCOUNTER, {})[r.DESCRIPTION] = float(r.VALUE)
    meds = pd.read_csv(csv / "medications.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
    mmap = {}
    for r in meds.itertuples():
        mmap.setdefault(r.PATIENT, []).append((str(r.START)[:10], str(r.STOP)[:10] if isinstance(r.STOP, str) else "9999", r.DESCRIPTION))
    encs = {}
    for r in enc.itertuples():
        encs.setdefault(r.PATIENT, []).append({"id": r.Id, "date": _d(r.START), "cls": r.ENCOUNTERCLASS, "desc": r.DESCRIPTION,
                                               "reason": r.REASONDESCRIPTION if isinstance(r.REASONDESCRIPTION, str) else None})
    popn = {"persons": persons, "encs": encs, "eobs": eobs, "meds": mmap}
    if cache:
        cache.write_bytes(pickle.dumps(popn))
    return popn


def augment(persons: dict, seed: int) -> None:
    """Occupation (from Synthea employment/education findings), work pattern, children, hobby, life events."""
    rng = random.Random(seed)
    higher = [o for o in OCCUPATIONS if o[1] == "higher"]
    hs = [o for o in OCCUPATIONS if o[1] == "hs"]
    for p in persons.values():
        age = (REF - p["birthdate"]).days // 365
        p["age_ref"] = age
        emp, edu = p["sdoh"].get("emp", ""), p["sdoh"].get("edu", "")
        is_higher = "higher" in edu.lower() or (not edu and p["income"] > 90000 and rng.random() < 0.6)
        occ = rng.choice(higher if is_higher else hs)
        p["occupation"], p["work_pattern"] = occ[0], None
        if age < 18:
            p.update(occ_status="child", occupation=None)
        elif age <= 22 and "Full-time" not in emp and rng.random() < 0.7:
            p.update(occ_status="student", occupation=None)
        elif age >= 67 or (age >= 60 and "labor force" in emp):
            p["occ_status"] = "retired"
        elif "Unemployed" in emp:
            p["occ_status"] = "unemployed"
        elif "labor force" in emp:
            p.update(occ_status="not working", occupation=None)
        else:
            p["occ_status"] = "working"
            if "Part-time" in emp and rng.random() < 0.6:
                p["work_pattern"] = "part-time"
            elif occ[2] and rng.random() < 0.5:
                p["work_pattern"] = rng.choice(["night shifts", "rotating shifts"])
            elif occ[0] == "truck driver" and rng.random() < 0.6:
                p["work_pattern"] = "long hauls"
            elif occ[1] == "higher" and rng.random() < 0.25:
                p["work_pattern"] = "remote"
        if p["marital"] is None:
            p["marital"] = "S" if age >= 18 else None
        births = [c for c in p["conds"] if c["desc"] == "Normal pregnancy (finding)" and c["stop"]]
        if age < 18:
            p["children"] = None
        else:
            probs = [.15, .18, .35, .2, .08, .03, .01] if p["marital"] in ("M", "D", "W") else [.6, .2, .13, .05, .02, 0, 0]
            if age < 30:
                probs = [probs[0] + .3] + [x * .6 for x in probs[1:]]
            k = rng.choices(range(7), weights=probs)[0]
            p["children"] = max(k, len(births)) if p["sex"] == "F" else k
        p["hobby"] = rng.choice(HOBBIES) if rng.random() < 0.6 and age >= 12 else None
        ev = []  # visible life events: fractures, concussions, heart attacks, childbirths, surgeries
        for c in p["conds"]:
            if any(k in c["desc"] for k in VISIBLE_INJURY):
                ev.append({"kind": "injury", "term": COND[c["desc"]]["term"], "lay": COND[c["desc"]]["lay"].replace("a ", "had a ", 1)
                           if c["desc"] != "Myocardial infarction (disorder)" else "had a heart attack", "year": c["start"].year,
                           "date": c["start"]})
        for b in births:
            ev.append({"kind": "childbirth", "term": "Vaginal delivery", "lay": "had my baby", "year": b["stop"].year, "date": b["stop"]})
        for t, dt in p["surgeries"].items():
            ev.append({"kind": "surgery", "term": t, "lay": f"had {'an' if t[0].lower() in 'aeiou' else 'a'} {t.lower()}",
                       "year": dt.year, "date": dt})
        p["events"] = sorted(ev, key=lambda e: e["date"])
        p["chronic"] = sorted({c["desc"]: c for c in p["conds"] if COND[c["desc"]]["kind"] == "chronic"}.values(),
                              key=lambda c: c["start"])


# ------------------------------------------------------------------ encounters
def build_encounter(p: dict, e: dict, popn: dict, rng: random.Random) -> dict:
    dt = e["date"]
    age = dt.year - p["birthdate"].year - ((dt.month, dt.day) < (p["birthdate"].month, p["birthdate"].day))
    chronic = [dict(c, **COND[c["desc"]]) for c in p["chronic"] if c["start"] <= dt and (c["stop"] is None or c["stop"] > dt)]
    reason = COND.get(e["reason"]) if e["reason"] else None
    if reason is None:
        here = [c for c in p["conds"] if c["enc"] == e["id"] and COND[c["desc"]]["kind"] != "chronic"]
        reason = COND[here[0]["desc"]] if here else None
    if reason is None and (e["cls"] == "wellness" or CHECKUP.search(e["desc"]) or not chronic):
        rk, lay, term, icd = "checkup", "check up", "Routine health examination", "Z00.129" if age < 18 else "Z00.00"
    elif reason is None:
        c = rng.choice(chronic)
        rk, lay, term, icd = "chronic", c["lay"], c["term"], c["icd"]
    else:
        rk = {"chronic": "chronic", "preg": "chronic"}.get(reason["kind"], "acute")
        lay, term, icd = reason["lay"], reason["term"], reason["icd"]
    ob = popn["eobs"].get(e["id"], {})
    vit = {VITALS[k]: ob[k] for k in VITALS if k in ob}
    for k in ("height", "weight", "bmi"):
        vit.setdefault(k, p.get(k))
    vit = {k: (round(x) if k in ("sbp", "dbp", "hr") else round(x, 1)) for k, x in vit.items() if x is not None and x == x}
    labs = []
    for k, (nm, sp, unit, dec) in LABS.items():
        if k in ob and nm not in {lab["name"] for lab in labs}:
            labs.append({"name": nm, "spoken": sp, "unit": unit, "value": f"{ob[k]:.{dec}f}"})
    meds, seen = [], set()
    for s, t, desc in popn["meds"].get(p["pid"], []):
        if s <= dt.isoformat() <= t and (m := _med(desc)) and m["drug"] not in seen:
            seen.add(m["drug"])
            meds.append(m)
    dosing = next((f"{m['drug']} dose" for m in meds if "enoxaparin" in m["drug"]), None) or \
        (f"{meds[0]['drug']} dose" if age < 18 and meds else None)
    icds = [icd] + [c["icd"] for c in chronic if c["icd"] != icd]
    return {"eid": None, "pid": p["pid"], "date": dt, "age": age, "reason_kind": rk, "reason_lay": lay, "reason_term": term,
            "primary_icd": icd, "icd": list(dict.fromkeys(icds))[:6], "chronic": chronic,
            "events": [x for x in p["events"] if x["date"] < dt], "vitals": vit, "labs": labs[:4], "meds": meds[:6],
            "weight_dosing": dosing, "travel": rng.choice(TRAVEL) if rng.random() < 0.1 else None,
            "race_word": RACE_WORD.get(p["race"])}


def sample_encounters(popn: dict, n_target: int, rng: random.Random) -> list[tuple[str, list]]:
    pids = sorted(pid for pid, es in popn["encs"].items() if es)
    rng.shuffle(pids)
    chosen, total = [], 0
    for pid in pids:
        if total >= n_target:
            break
        es = sorted(popn["encs"][pid], key=lambda e: e["date"])
        k = min(len(es), rng.choices([1, 2, 3], weights=[.72, .2, .08])[0], n_target - total)
        # weighted sampling without replacement: problem visits 3x more likely than routine check-ups
        keyed = sorted(es, key=lambda e: -rng.random() ** (1 / (1 if e["cls"] == "wellness" or CHECKUP.search(e["desc"]) else 3)))
        chosen.append((pid, sorted(keyed[:k], key=lambda e: e["date"])))
        total += k
    return chosen


def add_doppelgangers(persons: dict, popn: dict, pids: list[str], rng: random.Random, frac: float = 0.12) -> list[str]:
    """Deliberately hard cases: for a share of target patients, two decoys with the same sex, birth year, city, race,
    marital status, children, occupation and chronic-condition set (onset years shifted 0-2), different names and
    exact birth date. They enter the population, hence every register, like anybody else."""
    hard = rng.sample(sorted(pids), round(frac * len(pids)))
    firsts = {sx: [p["first"] for p in persons.values() if p["sex"] == sx] for sx in ("M", "F")}
    lasts = [p["last"] for p in persons.values()]

    def shift(d, dy):
        return d.replace(year=d.year + dy, day=min(d.day, 28))
    for pid in hard:
        src = persons[pid]
        for j in range(2):
            q = copy.deepcopy(src)
            q["pid"] = f"dup{j}-" + "".join(rng.choice("0123456789abcdef") for _ in range(12))
            q["first"], q["last"] = rng.choice(firsts[src["sex"]]), rng.choice(lasts)
            q["birthdate"] = date(src["birthdate"].year, rng.randint(1, 12), rng.randint(1, 28))
            q["age_ref"] = (REF - q["birthdate"]).days // 365
            for c in q["chronic"] + q["events"]:
                dy = rng.choice([-2, -1, 0, 1, 2])
                for k in ("start", "date"):
                    if k in c:
                        c[k] = shift(c[k], dy)
                if "year" in c:
                    c["year"] += dy
            for k, sd in (("height", 3), ("weight", 4)):
                if q.get(k):
                    q[k] = round(q[k] + rng.gauss(0, sd), 1)
            q["hobby"] = rng.choice(HOBBIES) if q.get("hobby") else None
            q["doppelganger_of"] = pid
            persons[q["pid"]] = q
            popn["encs"][q["pid"]] = [dict(e, date=e["date"] + timedelta(days=rng.randint(-60, 60)))
                                      for e in popn["encs"].get(pid, [])]
    return hard


# ------------------------------------------------------------------ attacker registers
def _jitter_interval(v: float, err_sd: float, half: float, rng: random.Random) -> tuple[float, float]:
    c = v + rng.gauss(0, err_sd)
    return round(c - half, 1), round(c + half, 1)


THREATS = {
    # what each attacker knows, with availability and error rates (documented in README)
    "A": dict(sex=.98, bd_full=0, by=.45, age_approx=.35, approx_err=3, city=.65, city_stale=.12, occ=.55, occ_stale=.10,
              race=.70, race_err=.05, marital=.30, marital_err=.05),
    "B": dict(sex=1, bd_full=.50, by=.35, age_approx=.15, approx_err=2, city=.90, city_stale=.05, occ=.85, occ_stale=.05,
              race=.95, race_err=.02, marital=.95, marital_err=.03, children=.90, children_err=.12, hobby=.75,
              hobby_err=.10, work=.70, body=.60, body_sd=(4, 6), body_half=(8, 12), events=.65, event_year_err=.25),
    "C": dict(sex=1, bd_full=.85, by=.13, age_approx=.02, approx_err=1, city=.95, city_stale=.03, occ=.90, occ_stale=.03,
              race=.97, race_err=.01, marital=.98, marital_err=.01, children=.97, children_err=.05, hobby=.80,
              hobby_err=.05, work=.80, body=.80, body_sd=(0.5, 1), body_half=(1, 2), events=.85, event_year_err=.10,
              icd=.60, icd_year=.70, visit=.50),
}


def make_register(persons: dict, popn: dict, threat: str, rng: random.Random) -> pd.DataFrame:
    t = THREATS[threat]
    cities = [p["city"] for p in persons.values()]
    titles = [o[0] for o in OCCUPATIONS]
    rows = []
    for p in persons.values():
        g = lambda k: rng.random() < t.get(k, 0)  # noqa: E731
        row = {"first": p["first"], "last": p["last"], "sex": p["sex"] if g("sex") else None}
        u = rng.random()
        if u < t["bd_full"]:
            row["birth_date"] = p["birthdate"].isoformat()
        elif u < t["bd_full"] + t["by"]:
            row["birth_year"] = p["birthdate"].year
        elif u < t["bd_full"] + t["by"] + t["age_approx"]:
            row["age_approx"] = p["age_ref"] + rng.randint(-t["approx_err"], t["approx_err"])
            row["age_tol"] = t["approx_err"]
        if g("city"):
            row["city"] = rng.choice(cities) if rng.random() < t["city_stale"] else p["city"]
        if p.get("occupation") and g("occ"):
            row["occupation"] = rng.choice(titles) if rng.random() < t["occ_stale"] else p["occupation"]
            row["occ_status"] = p["occ_status"]
        if g("race"):
            row["race"] = rng.choice(["white", "black", "asian", "hispanic"]) if rng.random() < t["race_err"] else p["race"]
        if p["marital"] and g("marital"):
            row["marital"] = rng.choice("MSDW") if rng.random() < t["marital_err"] else p["marital"]
        if p.get("children") is not None and g("children"):
            row["children"] = max(0, p["children"] + rng.choice([-1, 1])) if rng.random() < t["children_err"] else p["children"]
        if p.get("hobby") and g("hobby"):
            row["hobby"] = rng.choice(HOBBIES) if rng.random() < t["hobby_err"] else p["hobby"]
        if p.get("work_pattern") and g("work"):
            row["work_pattern"] = p["work_pattern"]
        if "body" in t and g("body"):
            for k, sd, half in (("height", t["body_sd"][0], t["body_half"][0]), ("weight", t["body_sd"][1], t["body_half"][1])):
                if p.get(k):
                    row[f"{k}_lo"], row[f"{k}_hi"] = _jitter_interval(p[k], sd, half, rng)
        ev = []
        for e in p["events"] if "events" in t else []:
            if rng.random() < t["events"]:
                ev.append(e["year"] + (rng.choice([-1, 1]) if rng.random() < t["event_year_err"] else 0))
        icd = []
        for c in p["chronic"] if "icd" in t else []:
            if rng.random() < t["icd"]:
                icd.append(COND[c["desc"]]["icd"][:3])
                if rng.random() < t["icd_year"]:
                    ev.append(c["start"].year + (rng.choice([-1, 1]) if rng.random() < 0.2 else 0))
        row["event_years"] = sorted(set(ev)) or None
        row["icd3"] = sorted(set(icd)) or None
        if "visit" in t and g("visit") and popn["encs"].get(p["pid"]):
            v = rng.choice(popn["encs"][p["pid"]])["date"]  # any visit in the window, released or not (symmetric)
            row["visit_ym"] = [f"{v.year}-{v.month:02d}"]
        row["pid"] = p["pid"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "key", [hashlib.sha256(f"{threat}|{rng.random()}".encode()).hexdigest()[:16] for _ in range(len(df))])
    return df.sample(frac=1, random_state=rng.randint(0, 2**31)).reset_index(drop=True)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthea", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--family", choices=["dev", "test"], required=True)
    ap.add_argument("--n-encounters", type=int, default=300)
    a = ap.parse_args(argv)
    csv, out = Path(a.synthea), Path(a.out)
    popn = load_population(csv, csv.parent / "population.pkl")
    persons = popn["persons"]
    augment(persons, seed=hash_seed(",".join(sorted(persons)[:50])))  # attributes depend on the population only
    rng = random.Random(a.seed)
    fam = 0 if a.family == "dev" else 1
    for d in ("private", "release_input", "attacker"):
        (out / d).mkdir(parents=True, exist_ok=True)
    recs, truth = [], []
    for pid, es in sample_encounters(popn, a.n_encounters, rng):
        p = persons[pid]
        for i, e in enumerate(es):
            enc = build_encounter(p, e, popn, rng)
            enc["eid"] = "".join(rng.choice("0123456789abcdef") for _ in range(12))
            dia, note, facts, exp = render_encounter(p, enc, fam, rng)
            recs.append({"encounter_id": enc["eid"], "dataset": "synthetic-reid", "dialogue": dia, "note": note,
                         "icd10_codes": [{"code": c, "description": ""} for c in enc["icd"]]})
            truth.append({"encounter_id": enc["eid"], "pid": pid, "visit_index": i, "date": enc["date"].isoformat(),
                          "age": enc["age"], "primary_icd": enc["primary_icd"], "reason_kind": enc["reason_kind"],
                          "facts": facts, "exposure": exp})
    hard = add_doppelgangers(persons, popn, sorted({t["pid"] for t in truth}), rng)
    order = list(range(len(recs)))
    rng.shuffle(order)
    with open(out / "release_input" / "encounters.jsonl", "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(recs[i]) + "\n")
    with open(out / "private" / "encounters_truth.jsonl", "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(truth[i]) + "\n")
    for t in THREATS:
        reg = make_register(persons, popn, t, random.Random(f"{a.seed}|{t}"))
        reg[["key", "pid"]].to_csv(out / "private" / f"keys_{t}.csv", index=False)
        reg.drop(columns=["pid"]).to_parquet(out / "attacker" / f"aux_{t}.parquet", index=False)
    cols = ["pid", "first", "last", "sex", "birthdate", "race", "marital", "city", "occupation", "occ_status", "work_pattern",
            "children", "hobby", "height", "weight", "bmi", "age_ref", "doppelganger_of"]
    pdf = pd.DataFrame([{c: p.get(c) for c in cols} | {"n_chronic": len(p["chronic"]), "n_events": len(p["events"])}
                        for p in persons.values()])
    with gzip.open(out / "private" / "persons.csv.gz", "wt", encoding="utf-8") as f:
        pdf.to_csv(f, index=False)
    man = {"synthea_csv": str(csv), "seed": a.seed, "family": a.family, "n_encounters": len(recs),
           "n_patients": len({t["pid"] for t in truth}), "population": len(persons), "reference_date": REF.isoformat(),
           "window": WINDOW, "threats": THREATS, "n_hard_targets": len(hard), "population_with_doppelgangers": len(persons)}
    (out / "private" / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("n_encounters", "n_patients", "population", "family", "seed")}))


def hash_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


if __name__ == "__main__":
    main()
