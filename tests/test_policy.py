"""Policy of 2026-10-17: DOB keeps the year, exact patient age 0-89 ("90+" for 90 or more), calendar years
kept, role-aware person placeholders. Every behaviour is paired with a dangerous counterexample.
All content is synthetic."""
from conftest import make_record


def _run(anonymizer, dialogue, note, eid="P1"):
    (out, decs), = anonymizer.anonymize_dataset([make_record(dialogue, note, eid=eid)])[0]
    return out, decs


# ---------------------------------------------------------------- A. date of birth keeps the year
def test_dob_keeps_year_in_numeric_iso_textual_and_compact_forms(anonymizer):
    note = ("DOB: 14/07/2013\nDate of birth: 2013-07-14\nBorn on March 22, 1985.\nDOB 07142013\n")
    out, _ = _run(anonymizer, "[doctor] hello .\n", note)
    lines = out["note"].split("\n")
    assert lines[0] == "DOB: [DOB] 2013" and lines[1] == "Date of birth: [DOB] 2013"
    assert lines[2] == "Born on [DOB] 1985." and lines[3] == "DOB [DOB] 2013"
    assert "14" not in out["note"] and "March" not in out["note"] and "07" not in out["note"]


def test_spoken_dob_answer_keeps_spoken_year(anonymizer):
    dlg = "[doctor] what's your date of birth ?\n[patient] march twenty second , nineteen eighty five .\n"
    out, decs = _run(anonymizer, dlg, "Follow up.\n")
    assert "[DOB] nineteen eighty five" in out["dialogue"] and "march" not in out["dialogue"]
    assert any(d["label"] == "DOB" for d in decs)


def test_dob_year_removed_when_patient_is_90_or_older(anonymizer):
    out, _ = _run(anonymizer, "[doctor] hello .\n", "The patient is a 93-year-old male. DOB: 02/11/1931.\n")
    assert "1931" not in out["note"] and "[DOB]" in out["note"] and "is a 90+-year-old male" in out["note"]


# ---------------------------------------------------------------- B. exact age 0-89, 90+ for 90 or more
def test_exact_ages_kept_up_to_89_in_note_and_dialogue(anonymizer):
    for age, spoken in ((6, "six"), (37, "thirty seven"), (65, "sixty five"), (89, "eighty nine")):
        out, _ = _run(anonymizer, f"[doctor] so you're {spoken} ?\n[patient] yes .\n", f"The patient is a {age}-year-old.\n")
        assert f"{age}-year-old" in out["note"] and spoken in out["dialogue"], age


def test_ages_of_90_and_above_are_top_coded_everywhere(anonymizer):
    out, _ = _run(anonymizer, "[doctor] okay , you're ninety two ?\n[patient] yes ninety two years old .\n",
                  "Mr. Quill is a 92-year-old male.\n")
    assert "92" not in out["note"] and "ninety two" not in out["dialogue"] and out["dialogue"].count("90+") == 2
    assert "Mr. [PATIENT_NAME] is a 90+-year-old male." in out["note"] or "[PATIENT_NAME] is a 90+-year-old male." in out["note"]
    assert "you're 90+ ?" in out["dialogue"] and "yes 90+ years old ." in out["dialogue"]


def test_age_exactly_90_and_89_boundaries(anonymizer):
    out, _ = _run(anonymizer, "[doctor] hello .\n", "Patient A is an 89-year-old man. His wife is 90 years old.\n")
    assert "an 89-year-old man" in out["note"]
    out, _ = _run(anonymizer, "[doctor] hello .\n", "The patient is a 90-year-old woman.\n")
    assert "a 90+-year-old woman" in out["note"]


def test_dataset_level_qi_step_does_not_widen_or_suppress_exact_age(anonymizer):
    recs = [make_record("[doctor] hello .\n", f"She is a {a}-year-old female.\n", eid=f"Q{i}") for i, a in enumerate([23, 51, 77])]
    results, stats = anonymizer.anonymize_dataset(recs)
    assert [f"{a}-year-old" in o["note"] for (o, _), a in zip(results, [23, 51, 77])] == [True] * 3
    assert stats["qi_verification"]["records_below_k"] == 3  # honest: exact ages keep these records unique


def test_relative_onset_age_dose_and_duration_are_not_patient_age(anonymizer):
    note = "Mother with breast cancer at age 92. Takes lisinopril 20 mg. Symptoms for 3 years. She is a 44-year-old woman.\n"
    out, decs = _run(anonymizer, "[doctor] hello .\n", note)
    assert "at age 92" in out["note"] and "20 mg" in out["note"] and "3 years" in out["note"] and "44-year-old" in out["note"]
    assert [d for d in decs if d["label"] == "AGE" and d["action"] != "keep"] == []


# ---------------------------------------------------------------- C. calendar years kept, identifiers are not
def test_year_next_to_clinician_name_is_kept(anonymizer):
    dlg = "[patient] i had surgery in 2018 with doctor jonathan smith and again in twenty twenty one .\n"
    note = "Knee arthroscopy in 2018 with Dr. Jonathan Smith. Appendectomy in March 2015. Diabetes since 2009.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert "in 2018 with [CLINICIAN_NAME]" in out["note"] and "in 2018 with" in out["dialogue"]
    assert "twenty twenty one" in out["dialogue"] and "[DATE] 2015" in out["note"] and "since 2009" in out["note"]
    assert "smith" not in (out["dialogue"] + out["note"]).lower() and "[ADDRESS]" not in out["note"]


def test_four_digit_identifiers_are_not_kept_as_years(anonymizer):
    dlg = "[patient] my pin is 2018 and i live at 2018 elm street .\n"
    note = "MRN: 20184411. Address: 1997 Oak Avenue. Hypertension since 2011.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert "2018" not in out["dialogue"] and "[ID]" in out["dialogue"] and "[ADDRESS]" in out["dialogue"]
    assert "20184411" not in out["note"] and "1997" not in out["note"] and "since 2011" in out["note"]


def test_year_like_numbers_after_identifier_cues_are_not_kept(anonymizer):
    dlg = "[patient] i'm in room 2014 , call extension 1987 .\n[doctor] your surgery was in 2014 .\n"
    out, _ = _run(anonymizer, dlg, "Admitted in 2014.\n")
    assert "room 2014" not in out["dialogue"] and "1987" not in out["dialogue"]
    assert "surgery was in 2014" in out["dialogue"] and "Admitted in 2014" in out["note"]


def test_large_doses_are_not_years_or_identifiers(anonymizer):
    dlg = "[doctor] take 2000 milligrams of vitamin c and 1000 units of vitamin d .\n"
    out, _ = _run(anonymizer, dlg, "Vitamin D 1000 units daily.\n")
    assert out["dialogue"] == dlg and "1000 units" in out["note"]


# ---------------------------------------------------------------- D. person placeholders
def test_relatives_and_clinicians_keep_distinct_role_placeholders(anonymizer):
    dlg = "[doctor] hi linnea , this is doctor brandt and your husband osvaldo is here .\n[patient] my sister ingrid called .\n"
    note = "Linnea Halvorsen is a 65-year-old female seen with Dr. Brandt. Husband Osvaldo and sister Ingrid are supportive.\n"
    out, _ = _run(anonymizer, dlg, note)
    both = out["dialogue"] + out["note"]
    assert "[PERSON" not in both and both.count("[PATIENT_NAME]") == 2 and both.count("[CLINICIAN_NAME]") == 2
    assert both.count("[RELATIVE_NAME_1]") == 2 and both.count("[RELATIVE_NAME_2]") == 2


def test_asr_fragment_does_not_split_the_patient_into_a_second_person(anonymizer):
    dlg = ("[doctor] the patient is tobias quarry .\n[doctor] mr . quarry is- is a 72-year-old male here for follow up .\n"
           "[doctor] good morning , mr . quarry .\n")
    out, _ = _run(anonymizer, dlg, "Tobias Quarry is a 72-year-old male.\n")
    assert "quarry" not in out["dialogue"].lower() and "[PERSON" not in out["dialogue"]


def test_clinical_eponyms_and_plural_terms_are_not_people(anonymizer):
    dlg = "[doctor] positive hawkins-kennedy and neer's test on the right , no arthralgias or myalgias .\n"
    note = "Musculoskeletal: No arthralgias, myalgias. Positive Hawkins-Kennedy and Neer's test.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert out["dialogue"] == dlg and out["note"] == note


def test_practice_named_after_a_profession_is_still_an_organization(anonymizer):
    dlg = "[patient] i got my glasses at optimum optometrists last year .\n"
    out, _ = _run(anonymizer, dlg, "Eye exam at Optimum Optometrists. No cataracts.\n")
    assert "optimum optometrists" not in out["dialogue"] and "Optimum Optometrists" not in out["note"]
    assert "No cataracts" in out["note"]


def test_real_person_with_eponym_surname_is_still_removed(anonymizer):
    out, _ = _run(anonymizer, "[doctor] hi mister hawkins , how are you ?\n", "Mr. Hawkins is a 70-year-old male.\n")
    assert "hawkins" not in (out["dialogue"] + out["note"]).lower()


# ---------------------------------------------------------------- evaluation integrity
def test_raw_and_policy_compliant_recall_are_reported_separately():
    # hand-calculated: 3 gold entities; the name is removed, the age is deliberately published,
    # the employer is missed -> raw recall 1/3, policy-compliant 2/3, one AGE kept by policy
    from medanon.evaluation.adversarial import _B
    from medanon.evaluation.scoring import score

    n = _B().phi("Ann Lee", "PERSON", "t").t(" is a ").phi("44", "AGE", "t").t("-year-old who works at ").phi("Zentrix", "ORGANIZATION", "t").t(".\n")
    rec = {"encounter_id": "S1", "dialogue": "", "note": n.text, "_gold": [dict(g, field="note") for g in n.gold], "_traps": []}
    decs = [{"field": "note", "start": 0, "end": 7, "action": "pseudonymize", "label": "PERSON"},
            {"field": "note", "start": 13, "end": 15, "action": "keep", "label": "AGE", "reason_code": "POLICY_KEEP"}]
    res = score([rec], [rec], [decs])
    assert (res["recall"]["k"], res["recall"]["n"]) == (1, 3)
    assert (res["policy_compliant_recall"]["k"], res["policy_compliant_recall"]["n"]) == (2, 3)
    assert res["kept_by_policy"] == {"AGE": 1} and res["recall_by"]["label:AGE"]["value"] == 0.0


def test_lab_intervals_keep_units_thresholds_and_trend(anonymizer):
    note = "RESULTS\n\nHemoglobin A1c 8.3%, improved from 9.6% in 2023. Glucose 250 mg/dL. EF 55-60%.\n"
    out, decs = _run(anonymizer, "[doctor] hello .\n", note)
    labs = [d for d in decs if d["label"] == "LAB_VALUE"]
    assert all(d["action"] == "generalize" for d in labs) and len(labs) >= 2
    # both values of the trend generalised to their clinical category: improvement still readable
    assert "A1c [8-9]%" in out["note"] and "from [9-10]%" in out["note"] and "mg/dL" in out["note"]
    assert "EF 55-60%" in out["note"] and "in 2023" in out["note"]  # ranges and years untouched
