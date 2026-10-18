"""Regressions from the review of synthetic development rounds 2 and 3 (2026-10-18), and the candidate
age-band policy. Every fix is paired with a dangerous counterexample. All content is invented."""
import copy
from contextlib import contextmanager

from conftest import make_record


def _run(anonymizer, dialogue, note, icd=None):
    (out, decs), = anonymizer.anonymize_dataset([make_record(dialogue, note, icd=icd)])[0]
    return out, decs


# ---------------------------------------------------------------- privacy: identifiers and surnames
def test_letter_prefixed_record_number_is_removed(anonymizer):
    out, _ = _run(anonymizer, "[patient] my mrn is qv-073319 .\n", "MRN QV-073319. Account no. 5521.\n")
    assert "073319" not in out["dialogue"] + out["note"] and "[ID]" in out["note"]


def test_ordinary_word_surname_after_first_name_is_removed(anonymizer):
    dlg = "[patient] Morning. My name is Tobin Ridge and I am here for my knee.\n"
    note = "Tobin Ridge is seen for knee pain. He walks on the north ridge trail daily.\n"
    out, _ = _run(anonymizer, dlg, note)
    both = out["dialogue"] + out["note"]
    assert "Tobin" not in both and "Ridge and I" not in out["dialogue"] and "Ridge is seen" not in out["note"]
    assert "north ridge trail" in out["note"]  # counterexample: the ordinary words stay where they are words


def test_month_after_first_name_starting_a_date_is_not_a_surname(anonymizer):
    out, _ = _run(anonymizer, "[doctor] hello .\n", "Seen by Dr. Ovelle. Follow up with Tobin March 9, 2027 if needed.\n")
    assert "Ovelle" not in out["note"] and "Tobin" not in out["note"]


# ---------------------------------------------------------------- clinical false positives
def test_derived_clinical_words_and_instruments_are_kept(anonymizer):
    dlg = ("[doctor] This is a supratherapeutic INR and lactational mastitis.\n"
           "[doctor] Dermatomal vesicles; Epworth Sleepiness Scale 16/24; positive Hawkins-Kennedy and Neer tests.\n")
    note = "ASSESSMENT\n\nSupratherapeutic INR. Lactational mastitis. Dermatomal vesicles. Epworth Sleepiness Scale 16/24.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert out["dialogue"] == dlg and out["note"] == note


def test_named_practice_is_still_removed(anonymizer):
    # counterexample to the instrument rule: an organisation is not a scale
    out, _ = _run(anonymizer, "[doctor] hello .\n", "Transferred from Quillmere Gate Family Practice. Wells score 2.\n")
    assert "Quillmere" not in out["note"] and "Wells score 2" in out["note"]


def test_units_without_a_number_and_pain_ranges_are_kept(anonymizer):
    dlg = "[patient] Do the letters in mIU/L matter? free T4 0.9 ng/dL and pain 2-7/10 .\n"
    out, _ = _run(anonymizer, dlg, "Pain 2-7/10. Seen 3/7/2021.\n")
    assert "mIU/L" in out["dialogue"] and "0.9 ng/dL" in out["dialogue"] and "2-7/10" in out["dialogue"] + out["note"]
    assert "3/7/2021" not in out["note"]  # a real date is still generalised


def test_staff_role_words_are_not_patient_occupations(anonymizer):
    note = "PLAN\n\nClinician to determine antibiotic need. Encounter completed. Support person present.\n"
    out, _ = _run(anonymizer, "[doctor] hello .\n", note)
    assert out["note"] == note


# ---------------------------------------------------------------- candidate age-band policy
@contextmanager
def _age_bands(anonymizer):
    saved = copy.deepcopy(anonymizer.cfg["policies"]["AGE"])
    anonymizer.cfg["policies"]["AGE"].clear()
    anonymizer.cfg["policies"]["AGE"].update({"action": "band", "exact_below": 18, "band_width": 10, "top_code": 90,
                                              "format": "[{band}]"})
    try:
        yield
    finally:
        anonymizer.cfg["policies"]["AGE"].clear()
        anonymizer.cfg["policies"]["AGE"].update(saved)


def test_age_bands_adults_banded_and_birth_year_removed(anonymizer):
    with _age_bands(anonymizer):
        out, _ = _run(anonymizer, "[doctor] so you're forty seven ?\n[patient] yes , born 03/14/1979 .\n",
                      "She is a 47-year-old woman. DOB: 03/14/1979. Mother had breast cancer at age 45.\n")
    assert "[40-49]-year-old" in out["note"] and "you're [40-49]" in out["dialogue"]
    assert "1979" not in out["dialogue"] + out["note"] and "[DOB]" in out["note"]
    assert "at age 45" in out["note"]  # family onset age is clinical, not the patient's age


def test_age_bands_children_exact_and_boundaries(anonymizer):
    with _age_bands(anonymizer):
        kid, _ = _run(anonymizer, "[doctor] hello .\n", "He is a 13-year-old boy. DOB: 05/02/2013.\n")
        b18, _ = _run(anonymizer, "[doctor] hello .\n", "She is an 18-year-old woman.\n")
        b89, _ = _run(anonymizer, "[doctor] hello .\n", "He is an 89-year-old man.\n")
        b93, _ = _run(anonymizer, "[doctor] hello .\n", "She is a 93-year-old woman. DOB: 01/02/1933.\n")
    assert "13-year-old" in kid["note"] and "[DOB] 2013" in kid["note"]
    assert "[18-19]-year-old" in b18["note"] and "[80-89]-year-old" in b89["note"]
    assert "90+-year-old" in b93["note"] and "1933" not in b93["note"]


def test_age_bands_keep_generic_relative_onset_ages(anonymizer):
    with _age_bands(anonymizer):
        out, _ = _run(anonymizer, "[patient] One close relative developed a related condition at age 63 .\n",
                      "Family history: relative developed gout at age 61. He is a 52-year-old man.\n")
    assert "at age 63" in out["dialogue"] and "at age 61" in out["note"] and "[50-59]-year-old" in out["note"]


def test_drug_combinations_and_analyte_values_are_kept(anonymizer):
    dlg = "[patient] Currently: Budesonide-formoterol controller and salbutamol rescue inhaler.\n"
    note = "RESULTS\n\ntTG-IgA 68 U/mL with normal total IgA; ferritin 8 ng/mL.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert out["dialogue"] == dlg and out["note"] == note


def test_medical_word_surname_removed_but_eponym_kept(anonymizer):
    dlg = "[patient] Hello. I am Hugo Stone and my wife came along.\n"
    note = "Hugo Stone presents with Bell's palsy on the left. History of a kidney stone.\n"
    out, _ = _run(anonymizer, dlg, note)
    assert "Stone and" not in out["dialogue"] and "Hugo" not in out["note"] and "Stone presents" not in out["note"]
    assert "Bell's palsy" in out["note"] and "kidney stone" in out["note"]


def test_workplace_called_x_and_rule_institution_labels(anonymizer):
    dlg = "[patient] The workplace is called Quillmere Publishing.\n[patient] The location field says Varrow Respiratory Clinic.\n"
    out, _ = _run(anonymizer, dlg, "Follow up.\n")
    assert "Quillmere" not in out["dialogue"] and "Varrow" not in out["dialogue"]
    assert "[PERSON" not in out["dialogue"]
