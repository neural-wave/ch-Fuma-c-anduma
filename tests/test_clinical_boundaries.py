"""Clinical boundary cases: numbers, negations, medications and terminology that look like PHI
patterns but must survive unchanged (synthetic)."""
import pytest

from conftest import make_record

UNCHANGED_NOTE_LINES = [
    "Blood pressure 128/76, heart rate 72.",
    "Pain is 7/10 in the right knee.",
    "Take 1/2 tablet of metoprolol 25 mg twice daily.",
    "G2P1 with history of gestational diabetes.",
    "COVID-19 vaccine series completed.",
    "Ejection fraction 55-60%.",
    "Follow up in 2-3 weeks.",
    "Grade 2/6 systolic murmur at the left sternal border.",
    "Lasix 40 mg daily and potassium 20 mEq daily.",
    "Denies fever, chills, or night sweats.",
    "No family history of colon cancer.",
    "Mother with breast cancer at age 45.",
    "Stage 3 chronic kidney disease, eGFR stable.",
    "Type 2 diabetes mellitus with diabetic neuropathy.",
    "Parkinson's disease, Bell's palsy and Hashimoto's thyroiditis.",
    "Status post appendectomy in childhood.",
]


@pytest.mark.parametrize("line", UNCHANGED_NOTE_LINES)
def test_clinical_line_unchanged(anonymizer, line):
    out, decs = anonymizer.anonymize(make_record("[doctor] okay .\n", "ASSESSMENT\n\n" + line + "\n", eid="CB1"))
    changed = [d for d in decs if d["action"] not in ("keep",) and d["field"] == "note" and d["label"] != "LAB_VALUE"]
    assert out["note"] == "ASSESSMENT\n\n" + line + "\n", (line, [(d["label"], d["detectors"]) for d in changed])


@pytest.mark.parametrize("turn", [
    "[doctor] any chest pain ?\n[patient] no , none at all .",
    "[patient] i don't smoke and i never drank .",
    "[doctor] take two tablets every six hours as needed .",
    "[patient] my mom had a stroke at sixty five .",
    "[doctor] your potassium was four point one which is normal .",
])
def test_clinical_dialogue_unchanged_except_labs(anonymizer, turn):
    out, decs = anonymizer.anonymize(make_record(turn + "\n", "Patient seen.", eid="CB2"))
    non_lab = [d for d in decs if d["action"] != "keep" and d["label"] != "LAB_VALUE"]
    assert not non_lab, [(d["label"], d["detectors"]) for d in non_lab]


def test_spoken_lab_value_never_matched_inside_a_word(anonymizer):
    dlg = "[patient] my creatinine , i was alone when they called , it was one point one .\n"
    out, decs = anonymizer.anonymize(make_record(dlg, "Creatinine 1.1 mg/dL.\n"))
    assert "alone" in out["dialogue"]
    labs = [d for d in decs if d["label"] == "LAB_VALUE" and d["field"] == "dialogue"]
    assert all(not dlg[d["start"] - 1].isalnum() for d in labs)


# ---------------------------------------------------------------- precision fixes (holdout review)
def test_common_word_surname_after_first_name_is_removed(anonymizer):
    note = "HISTORY OF PRESENT ILLNESS\n\nMichelle King is a pleasant 44-year-old female. Green sputum was not reported.\n"
    out, _ = anonymizer.anonymize(make_record("[doctor] hi michelle .\n", note))
    assert "Michelle" not in out["note"] and "King" not in out["note"]
    assert "Green sputum" in out["note"]  # a capitalised common word that is not after a first name stays


def test_dosed_brand_drug_is_not_an_organization(anonymizer):
    dlg = "[doctor] i will write you a prescription for ultram , take ultram fifty milligrams once a day .\n"
    note = "PLAN\n\nPrescription for Ultram 50 mg once per day.\n"
    out, _ = anonymizer.anonymize(make_record(dlg, note))
    assert out["dialogue"].count("ultram") == 2 and "Ultram 50 mg" in out["note"]


def test_drug_named_person_still_removed_next_to_the_drug(anonymizer):
    dlg = "[patient] my friend lyrica told me to come in .\n[doctor] keep taking lyrica seventy five milligrams twice a day .\n"
    out, _ = anonymizer.anonymize(make_record(dlg, "Lyrica 75 mg twice daily.\n"))
    assert "friend lyrica" not in out["dialogue"] and "lyrica seventy five" in out["dialogue"]


def test_ordinals_before_nouns_are_not_dates(anonymizer):
    dlg = "[patient] i was fine after the second crash but after the third one i had surgery .\n[doctor] see you on the fifth .\n"
    out, _ = anonymizer.anonymize(make_record(dlg, "Follow up.\n"))
    assert "the second crash" in out["dialogue"] and "the third one" in out["dialogue"] and "the fifth" not in out["dialogue"]


def test_test_names_asr_contractions_and_ligaments_untouched(anonymizer):
    dlg = ("[doctor] i wan na check a chem-12 blood test , and i'm worried about your acl .\n"
           "[patient] i wan na get back to running .\n")
    out, _ = anonymizer.anonymize(make_record(dlg, "Chem-12 ordered. Possible ACL tear.\n"))
    assert out["dialogue"] == dlg and "Chem-12" in out["note"] and "ACL tear" in out["note"]
