"""Regressions found in the review of the 25 synthetic development encounters (2026-10-17).
Each fix is paired with a dangerous counterexample. All content is invented."""
from conftest import make_record


def _run(anonymizer, dialogue, note, icd=None):
    (out, decs), = anonymizer.anonymize_dataset([make_record(dialogue, note, icd=icd)])[0]
    return out, decs


# ---------------------------------------------------------------- clinical false positives
def test_clinical_signs_and_eponym_adjectives_are_kept(anonymizer):
    dlg = ("[doctor] We have mild resting tremor and bradykinesia on the right.\n"
           "[doctor] My working assessment is Parkinsonian features; we can try canalith repositioning.\n")
    note = "PHYSICAL EXAM\n\nMild resting tremor and bradykinesia on right.\n\nASSESSMENT\n\nParkinsonian features.\n"
    out, _ = _run(anonymizer, dlg, note, icd=[{"code": "R25.1", "description": "Tremor, unspecified"}])
    assert out["dialogue"] == dlg and out["note"] == note


def test_clinical_verbs_and_instructions_are_not_organisations(anonymizer):
    dlg = "[doctor] For the plan, Reassess for imaging based on exam.\n[doctor] Before you go: Seek urgent care for chest pain.\n"
    note = "PLAN\n\nReassess for imaging based on exam.\n\nINSTRUCTIONS\n\nSeek urgent care for chest pain.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert out["dialogue"] == dlg and out["note"] == note


def test_named_urgent_care_and_hospital_are_still_removed(anonymizer):
    # counterexample: a capitalised institution name with the same head words
    note = "Seen at Quillmere Urgent Care last week, then admitted to Varrowfield Hospital.\n"
    out, _ = _run(anonymizer, "[doctor] hello .\n", note)
    assert "Quillmere" not in out["note"] and "Varrowfield" not in out["note"]


def test_partial_tokens_and_units_are_not_replaced(anonymizer):
    dlg = "[doctor] We have SpO2 98% room air.\n[doctor] Your TSH was 8.1 mIU/L, free T4 normal.\n"
    note = "RESULTS\n\nSpO2 98% room air. TSH 8.1 mIU/L.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert "SpO2 98%" in out["dialogue"] and "SpO2 98%" in out["note"]
    assert "mIU/L" in out["dialogue"] and "mIU/L" in out["note"] and "[PERSON]" not in out["dialogue"] + out["note"]


def test_name_glued_to_digits_elsewhere_still_removed(anonymizer):
    # counterexample: the boundary fix must not let a real name through
    out, _ = _run(anonymizer, "[doctor] hi mister quarrington , how are you ?\n", "Mr. Quarrington is a 61-year-old male.\n")
    assert "quarrington" not in (out["dialogue"] + out["note"]).lower()


# ---------------------------------------------------------------- punctuation and labels
def test_replacements_keep_sentence_punctuation(anonymizer):
    dlg = "[patient] It is 18 Quillmere Lane, East Varrow. No changes.\n[patient] Email me at jq@varrowmail.com, please.\n"
    out, _ = _run(anonymizer, dlg, "Follow up.\n")
    assert "Quillmere" not in out["dialogue"] and "varrowmail" not in out["dialogue"]
    assert "]. No changes." in out["dialogue"] and "], please." in out["dialogue"]


def test_abbreviation_period_inside_a_sentence_is_kept(anonymizer):
    out, _ = _run(anonymizer, "[patient] I live at 12 Main St. in the town centre .\n", "Follow up.\n")
    assert "12 Main" not in out["dialogue"] and "]. in" not in out["dialogue"]


def test_business_called_x_is_an_organisation_but_son_called_x_is_a_person(anonymizer):
    dlg = ("[patient] I work at a small company. It is called Bramblewick Craft Services.\n"
           "[patient] My son is called Tobias Quarrington and he drove me.\n")
    out, _ = _run(anonymizer, dlg, "Works at Bramblewick Craft Services.\n")
    assert "Bramblewick" not in out["dialogue"] + out["note"] and "Quarrington" not in out["dialogue"]
    assert "It is called [EMPLOYER]." in out["dialogue"] and "My son is called [" in out["dialogue"]
    assert "called [PERSON]." not in out["dialogue"]


def test_phone_read_out_after_question_is_contact(anonymizer):
    dlg = ("[doctor] Let me check the contact number listed for you. Can you say it out loud?\n"
           "[patient] Yes, it is two zero two five five five zero one two two.\n")
    out, _ = _run(anonymizer, dlg, "Follow up.\n")
    assert "[CONTACT]" in out["dialogue"] and "five five five" not in out["dialogue"]


def test_generic_workplace_kept_named_one_removed(anonymizer):
    note = "SOCIAL HISTORY\n\nWorks at a pet-supply store; previously at Fernwick Outfitters.\n"
    out, _ = _run(anonymizer, "[doctor] hello .\n", note)
    assert "a pet-supply store" in out["note"] and "Fernwick" not in out["note"]


def test_named_clinic_is_a_healthcare_facility(anonymizer):
    out, _ = _run(anonymizer, "[doctor] We are at Fernhollow Community Clinic today.\n", "Seen at Fernhollow Community Clinic.\n")
    assert "Fernhollow" not in out["dialogue"] + out["note"] and "[HEALTHCARE_FACILITY]" in out["note"]


# ---------------------------------------------------------------- lab ranges and NSAID brands
def test_lab_ranges_are_not_dates_and_stay_intact(anonymizer):
    dlg = "[doctor] your hemoglobin was between 11 and 12 , hemoglobin 11-12 .\n"
    note = "RESULTS\n\nHemoglobin 11.2-12.4 g/dL. Visit 03-14-2022.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert "between 11 and 12" in out["dialogue"] and "11-12" in out["dialogue"]
    assert "Hemoglobin 11.2-12.4 g/dL" in out["note"] and "03-14" not in out["note"]  # a real date is still removed


def test_nsaid_brand_names_are_kept(anonymizer):
    note = "MEDICATIONS\n\nToradol given in clinic. Mobic 15 mg daily. Feldene stopped.\n"
    out, _ = _run(anonymizer, "[doctor] we gave you toradol today .\n", note)
    assert "Toradol given" in out["note"] and "Mobic 15 mg" in out["note"] and "toradol today" in out["dialogue"]
