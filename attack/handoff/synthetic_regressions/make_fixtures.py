"""Writes encounters.jsonl + expectations.json: minimal, fully synthetic probes (invented people,
places and values) for each weakness the attacker measured or the other attacker suggested,
plus clinical counterexamples that must survive anonymization unchanged.

    python make_fixtures.py            # regenerates both files next to this script
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ICD = [{"code": "I10", "description": "Essential (primary) hypertension"}]

# (encounter_id, finding, dialogue, note, assertions)
# assertion: (kind, regex, expect) with kind privacy|clinical, expect absent|present (case-insensitive,
# searched in dialogue + note of the anonymized record).
PROBES = [
    ("SYN-LANDMARK-01", "V1_landmark_context_leak",
     "[doctor] hi frank , how was the trip ?\n[patient] great . i went to oregon last weekend to hike around mount hood , "
     "near the timberline lodge .\n[doctor] good . any chest pain on the climb ?\n[patient] no .",
     "HISTORY OF PRESENT ILLNESS\nFrank Dawson is a 52-year-old male with hypertension. He recently hiked near Mount Hood "
     "in Oregon without chest pain.",
     [("privacy", r"oregon", "absent"), ("privacy", r"mount hood", "absent"), ("privacy", r"timberline", "absent"),
      ("clinical", r"hik", "present"), ("clinical", r"chest pain", "present"), ("clinical", r"hypertension", "present")]),
    ("SYN-AGE-01", "V2_exact_adult_age",
     "[doctor] so carla is a 47-year-old female here for blood pressure follow up .\n[patient] yes .\n"
     "[doctor] you smoked for 20 years , right ?\n[patient] yes , quit last year .",
     "HISTORY OF PRESENT ILLNESS\nCarla Mendes is a 47-year-old female presenting for hypertension follow-up. "
     "She has a 20 pack-year smoking history.",
     [("privacy", r"\b47[- ]?year", "absent"),
      ("clinical", r"20 years", "present"), ("clinical", r"20 pack-year", "present")]),
    ("SYN-AGE-02", "L1_bare_age_answer",
     "[doctor] and how old are you now ?\n[patient] 54 .\n[doctor] okay , and any headaches ?\n[patient] no .",
     "HISTORY OF PRESENT ILLNESS\nThe patient is here for hypertension follow-up. Denies headaches.",
     [("privacy", r"\[patient\] 54\b", "absent"), ("clinical", r"headaches", "present")]),
    ("SYN-DOB-01", "L2_birth_year_4_and_2_digit",
     "[doctor] can you confirm your date of birth ?\n[patient] march 3rd , 1971 .\n[doctor] thanks . and your husband ?\n"
     "[patient] he was born in '68 .",
     "HISTORY OF PRESENT ILLNESS\nPatient (DOB 03/03/71) presents for hypertension follow-up.",
     [("privacy", r"1971", "absent"), ("privacy", r"03/03/71", "absent"), ("privacy", r"\[DOB\]\s*'?71\b", "absent"),
      ("privacy", r"'68\b", "absent"), ("clinical", r"hypertension", "present")]),
    ("SYN-DOB-02", "L2_birth_year_with_adult_age_stated",
     "[doctor] so this is a 52-year-old male . date of birth is march 3rd , 1974 ?\n[patient] yes .",
     "HISTORY OF PRESENT ILLNESS\nThe patient is a 52-year-old male, DOB 03/03/1974, here for hypertension.",
     [("privacy", r"1974", "absent"), ("privacy", r"\b52[- ]?year", "absent"), ("clinical", r"hypertension", "present")]),
    ("SYN-BP-02", "L3_blood_pressure_lowercase_variants",
     "[doctor] your bp is 150 over 95 today .\n[patient] wow .\n[doctor] the bp was 138/86 last visit , and your "
     "pressure of one forty over ninety at the pharmacy is high too .",
     "VITALS\nbp 150/95.\n\nASSESSMENT\nUncontrolled hypertension (bp 138/86 last visit).",
     [("clinical", r"bp is 150 over 95", "present"), ("clinical", r"bp was 138/86", "present"),
      ("clinical", r"one forty over ninety", "present"), ("clinical", r"bp 150/95", "present"),
      ("clinical", r"bp 138/86", "present")]),
    ("SYN-OVER-03", "V3_possessive_and_product_overredaction",
     "[doctor] your grips are equal . neurovascular's intact . try some icy hot or biofreeze on it .\n[patient] okay .",
     "PHYSICAL EXAMINATION\nGrips equal. Neurovascular status intact.\n\nPLAN\nTopical Icy Hot or Biofreeze.",
     [("clinical", r"neurovascular's intact", "present"), ("clinical", r"Neurovascular status intact", "present"),
      ("clinical", r"icy hot", "present"), ("clinical", r"biofreeze on it", "present"), ("clinical", r"Icy Hot or Biofreeze", "present")]),
    ("SYN-OVER-04", "V3_spoken_exam_and_symptom_overredaction",
     "[doctor] let me take a listen to your heart and lungs okay .\n[patient] okay .\n[doctor] i do notice some crackles in your lungs "
     "bilaterally . and if he gets a barky cough at night , give him the steroid . i'm gon na place a referral to cardiology .",
     "PHYSICAL EXAMINATION\nCrackles in the lungs bilaterally.\n\nPLAN\nReturn precautions for barky cough. Referral to cardiology placed.",
     [("clinical", r"heart and lungs", "present"), ("clinical", r"crackles in your lungs", "present"),
      ("clinical", r"barky cough at night", "present"), ("clinical", r"place a referral", "present"),
      ("clinical", r"Crackles in the lungs", "present"), ("clinical", r"barky cough\. Referral", "present")]),
    ("SYN-BP-01", "L3_blood_pressure_mislabelled",
     "[doctor] hey , dragon , show me the vital signs . your blood pressure today is 142 over 88 , and your heart rate is 72 .\n"
     "[patient] okay .\n[doctor] last month it was 128/82 at home .",
     "VITALS\nBP: 142/88 mmHg. HR: 72 bpm.\n\nASSESSMENT AND PLAN\nHypertension, above goal. Home BP 128/82.",
     [("clinical", r"142 over 88", "present"), ("clinical", r"142/88", "present"), ("clinical", r"128/82", "present"),
      ("clinical", r"\b72\b", "present"), ("clinical", r"blood pressure today is \[(?:ORGANIZATION|LOCATION|PERSON)", "absent")]),
    ("SYN-OVER-01", "V3_clinical_term_overredaction",
     "[doctor] did you get your covid 19 vaccine ?\n[patient] yes , both doses .\n[doctor] let me listen to your heart and lungs . "
     "your lungs are clear . you are neurovascularly intact .\n[doctor] use some biofreeze on the neck , and we will order "
     "some autoimmune labs .",
     "HISTORY OF PRESENT ILLNESS\nFully vaccinated against COVID-19.\n\nPHYSICAL EXAMINATION\nLungs clear to auscultation. "
     "Neurovascular: intact distally.\n\nPLAN\nBiofreeze topically. Autoimmune labs ordered.",
     [("clinical", r"covid", "present"), ("clinical", r"heart and lungs", "present"), ("clinical", r"lungs are clear", "present"),
      ("clinical", r"neurovascular", "present"), ("clinical", r"biofreeze", "present"), ("clinical", r"autoimmune labs", "present")]),
    ("SYN-OVER-02", "V3_name_swallows_clinical_word",
     "[doctor] okay frank , your lungs are clear and your heart sounds regular .\n[patient] good .",
     "PHYSICAL EXAMINATION\nLungs clear bilaterally. Heart regular rate and rhythm.",
     [("privacy", r"\bfrank\b", "absent"), ("clinical", r"your lungs are clear", "present")]),
    ("SYN-KEEP-01", "C1_medications_labs_chronology",
     "[doctor] your ejection fraction was 45 % and your a1c came back at 8.2 .\n[patient] okay .\n[doctor] increase metformin "
     "to 1000 mg twice daily and lisinopril 20 milligrams once a day . you had back surgery about 10 years ago , and "
     "this pain started four days ago .",
     "RESULTS\nEchocardiogram: ejection fraction 45%. Hemoglobin A1c 8.2.\n\nPLAN\nIncrease metformin to 1000 mg twice daily. "
     "Lisinopril 20 mg daily. History of lumbar surgery 10 years ago; current pain for 4 days.",
     [("clinical", r"metformin to 1000 mg twice daily", "present"), ("clinical", r"lisinopril 20 (?:mg|milligrams)", "present"),
      ("clinical", r"about 10 years ago", "present"), ("clinical", r"four days ago", "present"),
      ("clinical", r"ejection fraction (?:was |of )?(?:\[?\d)", "present"), ("clinical", r"a1c (?:came back at |)\[?\d", "present")]),
]


def main() -> None:
    with open(HERE / "encounters.jsonl", "w", encoding="utf-8") as f:
        for eid, _finding, dialogue, note, _ in PROBES:
            f.write(json.dumps({"encounter_id": eid, "dataset": "synthetic", "dialogue": dialogue, "note": note,
                                "icd10_codes": ICD}) + "\n")
    expectations = [{"encounter_id": eid, "finding": finding, "kind": kind, "regex": rx, "expect": expect}
                    for eid, finding, _, _, asserts in PROBES for kind, rx, expect in asserts]
    (HERE / "expectations.json").write_text(json.dumps(expectations, indent=1) + "\n", encoding="utf-8")
    print(f"{len(PROBES)} encounters, {len(expectations)} assertions")


if __name__ == "__main__":
    main()
