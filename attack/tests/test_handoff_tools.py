"""Audit, export and synthetic-regression tools used by the MedAnon handoff."""
import json
import math
import subprocess
import sys
from pathlib import Path

from attack.audit_masks import classify, replacements, residue
from attack.export_predictions import export

HANDOFF = Path(__file__).resolve().parents[1] / "handoff" / "synthetic_regressions"


def test_alignment_classifies_overredaction_and_identifiers():
    reps = replacements("let me listen to your heart and [PERSON] . mrn is [ID] . hi [PATIENT_NAME] .",
                        "let me listen to your heart and lungs . mrn is 123546 . hi frank dawson .")
    assert [r["category"] for r in reps] == ["clinical_term", "identifier", "identifier"]
    assert classify("142 over 88") == "measurement_bp" and classify("march 3rd") == "date_time"


def test_residue_scans():
    assert residue("[doctor] how old was your mother ?\n[patient] mom was 40 .")["age_question_answers"] == []
    r = residue("[doctor] how old are you ?\n[patient] 54 .\na 47-year-old male , smoked for 20 years , "
                "a [40-49]-year-old sister , born in '68 , dob 1971 , 3-year-old son")
    assert r["exact_adult_ages"] == ["47"] and r["age_question_answers"] == ["54"]
    assert r["birth_years_2digit"] == ["68"] and r["birth_years_4digit"] == ["1971"]
    assert residue("date of birth is [DOB] 1982 .")["birth_years_4digit"] == ["1982"]
    assert residue("Patient (DOB [DOB] 71) presents")["birth_years_2digit"] == ["71"]
    exact, band = math.log2(1 / 0.0128), 0  # one-year cohort ~6.3 bits; the band is cheaper
    assert 6.0 < r["age_bits_max"] < 6.6 and exact > band


def test_export_drops_borrowed_facts_and_truth():
    row = {"encounter_id": "E", "risk": 0.9, "lambda": 0.1, "bits": 30.0, "failed": False,
           "attributes": [{"field": "age", "value": "50", "evidence": "50-year-old", "category": "age", "p": 0.0128,
                           "p_source": "census", "bits": 6.3, "counted": True},
                          {"field": "state_or_region", "value": "Washington", "evidence": "x", "category": "state",
                           "p": 0.023, "p_source": "census", "bits": 5.4, "counted": True, "from_original": True}],
           "recovery": [{"placeholder": "[US_STATE]", "type": "state_or_region", "guess": "Washington",
                         "confidence": "high", "truth": ["washington"], "correct": True, "reasoning": "r"}]}
    out = export(row)
    assert [q["value"] for q in out["quasi_identifiers"]] == ["50"]
    assert abs(out["bits"] - math.log2(1 / 0.0128)) < 0.2 and out["risk"] < 1e-5
    assert set(out["recovery_guesses"][0]) == {"placeholder", "type", "guess", "confidence"}


def test_synthetic_checker_on_raw_input(tmp_path):
    """Raw input leaks every identifier and keeps every clinical fact: privacy all fail, clinical all pass."""
    res = tmp_path / "r.json"
    proc = subprocess.run([sys.executable, str(HANDOFF / "check_regressions.py"), str(HANDOFF / "encounters.jsonl"),
                           "--json", str(res)], capture_output=True, text=True)
    results = json.loads(res.read_text())
    assert proc.returncode == 0
    # residue-form checks ("[DOB] 71") pass trivially on raw text, which has no placeholders
    assert not any(r["pass"] for r in results if r["kind"] == "privacy" and "\\[DOB\\]" not in r["regex"])
    assert all(r["pass"] for r in results if r["kind"] == "clinical")
