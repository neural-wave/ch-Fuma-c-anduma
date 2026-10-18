"""Hardening fixes (synthetic, fictional encounters) under the recommended configs/age_bands.yaml profile.

Set MEDANON_TEST_FAST=1 to run without the transformer detectors (rules + spaCy only, quick local check).
"""
import os
from pathlib import Path

import pytest

from medanon import Anonymizer, load_config
from medanon.manifest import _NOT_CODE
from medanon.transform import birth_year

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def band():
    over = {"ml": {"enabled": False}, "extra_detectors": []} if os.environ.get("MEDANON_TEST_FAST") else None
    return Anonymizer(load_config(ROOT / "configs" / "age_bands.yaml", over))


def run(anon, dialogue="[doctor] hello .\n[patient] hi .", note="Patient seen today.", icd=None):
    rec = {"encounter_id": "H1", "dataset": "aci", "dialogue": dialogue, "note": note,
           "icd10_codes": icd or [{"code": "I10", "description": "Essential (primary) hypertension"}]}
    out, _ = anon.anonymize(rec)
    return out["dialogue"], out["note"]


# ------------------------------------------------------------------ A. vital signs and section headers
def test_vital_signs_survive(band):
    _, note = run(band, note="VITALS\nBP 114/71, HR 81, Height 170 cm.\nVital Signs: BP: 130/85, RR 16, SpO2 97%.\n"
                             "Blood pressure 128/82 mmHg, temperature 98.6 F.\nHe works at Pinebrook Logistics.")
    for keep in ("BP 114/71", "HR 81", "BP: 130/85", "RR 16", "SpO2 97%", "Blood pressure 128/82 mmHg", "98.6"):
        assert keep in note, keep
    assert "Pinebrook" not in note  # a genuine employer is still anonymised


def test_section_header_kept(band):
    _, note = run(band, note="SOCIAL HX\nWorks in an office. Denies tobacco use.\n\nREVIEW OF SYSTEMS\nNegative.")
    assert "SOCIAL HX" in note and "REVIEW OF SYSTEMS" in note


# ------------------------------------------------------------------ V2. clinical words never become identifiers
def test_clinical_words_not_masked(band):
    dia, _ = run(band, dialogue="[doctor] hi mrs. kowalczyk .\n[doctor] your lungs are clear and neurovascular's intact .\n"
                                "[doctor] use some biofreeze on the neck and i'll put in a referral .\n"
                                "[patient] my kidney doctor said the same thing .")
    for keep in ("lungs", "neurovascular", "biofreeze", "referral", "kidney doctor"):
        assert keep in dia, keep
    assert "kowalczyk" not in dia


# ------------------------------------------------------------------ B. adult birth year, independent of a detected age
def test_adult_birth_year_removed_without_stated_age(band):
    _, note = run(band, note="DOB: 03/04/1978\nThe patient presents for follow-up.")
    assert "1978" not in note and "[DOB]" in note


def test_two_digit_birth_year_and_filler_answer(band):
    dia, _ = run(band, dialogue="[doctor] just to confirm , what's your date of birth ?\n[patient] okay , 3/4/78\n"
                                "[doctor] thanks .")
    assert "78" not in dia


def test_child_birth_year_kept_and_clinical_years_kept(band):
    _, note = run(band, note="DOB: 05/06/2015\nShe is a 11-year-old girl.\nAsthma diagnosed in 2019. "
                             "Tonsillectomy on 03/12/2021.")
    assert "2015" in note  # paediatric policy keeps the birth year
    assert "2019" in note and "[DATE] 2021" in note  # clinical chronology untouched
    assert birth_year("3/4/78") == 1978 and birth_year("12/01/09") == 2009 and birth_year("[DOB]") is None


# ------------------------------------------------------------------ C. bare answers to age questions
def test_bare_age_answer_banded(band):
    dia, _ = run(band, dialogue="[doctor] and how old are you now ?\n[patient] uh , 49 .\n"
                                "[doctor] you're how old now ?\n[patient] 52\n[doctor] okay .")
    assert "uh , 49" not in dia and "[patient] 52" not in dia and "[40-49]" in dia


def test_numbers_that_are_not_the_patients_age(band):
    dia, _ = run(band, dialogue="[doctor] how old is your son ?\n[patient] 12 .\n"
                                "[doctor] how bad is the pain from one to ten ?\n[patient] 7 .\n"
                                "[doctor] what was your blood pressure at home ?\n[patient] 130 over 80 .\n"
                                "[doctor] how long have you had it ?\n[patient] 3 weeks .")
    for keep in ("12", "7", "130 over 80", "3 weeks"):
        assert keep in dia, keep


# ------------------------------------------------------------------ D. compact household size
def test_compact_family_size(band):
    dia, note = run(band, dialogue="[patient] i live with my wife and our three kids .",
                    note="SOCIAL HISTORY\nMarried, 4 children. Works in sales.")
    assert "4 children" not in note and "[3+]" in note
    assert "three kids" not in dia


def test_family_counts_that_stay_exact(band):
    _, note = run(band, note="SOCIAL HISTORY\nMarried, 2 children.\nFAMILY HISTORY\nHer three brothers all have hemophilia.")
    assert "2 children" in note and "three brothers" in note


# ------------------------------------------------------------------ A1. reproducible code hash
def test_code_hash_ignores_virtualenvs():
    assert {"env", ".venv", "venv", "site-packages", "__pycache__"} <= _NOT_CODE


# ------------------------------------------------------------------ E. towns named after a residence cue
def test_town_after_residence_cue(band):
    dia, note = run(band, dialogue="[patient] we moved to cambridge last year .",
                    note="SOCIAL HISTORY\nDivorced. Lives in Cambridge. Drinks alcohol socially. Lives in Fall River.")
    assert "Cambridge" not in note and "Fall River" not in note and "cambridge" not in dia


def test_common_words_not_places_without_cue(band):
    _, note = run(band, note="Seen in urgent care provider network. Follow up on the West Coast trip in two weeks.")
    assert "urgent care" in note and "two weeks" in note


# ------------------------------------------------------------------ V4. named landmarks next to a masked place
def test_named_landmark_masked(band):
    dia, _ = run(band, dialogue="[patient] i went to oregon last weekend to hike around mount hood .\n"
                                "[doctor] did you mount the brace on the left knee ? any trouble at lake level ?")
    assert "mount hood" not in dia and "hood" not in dia
    assert "mount the brace" in dia and "lake level" in dia


# ------------------------------------------------------------------ independent red-team probes (teammate handoff)
def test_redteam_handoff_probes(band):
    import json
    import re
    fx = ROOT / "tests" / "fixtures" / "redteam_handoff"
    out = {}
    for line in open(fx / "encounters.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        o, _ = band.anonymize(rec)
        out[rec["encounter_id"]] = o["dialogue"] + "\n" + o["note"]
    failed = [(e["encounter_id"], e["kind"], e["regex"]) for e in json.loads((fx / "expectations.json").read_text(encoding="utf-8"))
              if (re.search(e["regex"], out[e["encounter_id"]], re.I) is not None) != (e["expect"] == "present")]
    assert not failed, failed
