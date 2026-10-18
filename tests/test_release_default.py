"""The release default must be the age-band policy, identical to an explicit --config configs/age_bands.yaml."""
from pathlib import Path

from medanon import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_policy_is_age_bands():
    assert load_config()["policies"] == load_config(ROOT / "configs" / "age_bands.yaml")["policies"]
    assert load_config(ROOT / "configs" / "default.yaml")["policies"]["AGE"] == load_config()["policies"]["AGE"]
    assert load_config()["policies"]["AGE"]["action"] == "band"


def test_legacy_exact_age_profile_still_available():
    assert load_config(ROOT / "configs" / "exact_age.yaml")["policies"]["AGE"]["action"] == "exact"


def test_implicit_and_explicit_age_bands_give_identical_output():
    from medanon import Anonymizer
    over = {"ml": {"enabled": False}, "extra_detectors": []}  # rules + spaCy: the policy step is what is compared
    rec = {"encounter_id": "R1", "dataset": "aci", "icd10_codes": [{"code": "I10", "description": "Essential hypertension"}],
           "dialogue": "[doctor] how old are you now ?\n[patient] 47 .\n[doctor] date of birth ?\n[patient] okay , 3/4/78",
           "note": "DOB: 03/04/1978\nThe patient is a 47-year-old man. Married, 4 children. BP 128/82."}
    a = Anonymizer(load_config(None, over)).anonymize(rec)[0]
    b = Anonymizer(load_config(ROOT / "configs" / "age_bands.yaml", over)).anonymize(rec)[0]
    assert a == b
    assert "47" not in a["note"] and "1978" not in a["note"] and "[40-49]" in a["note"] and "BP 128/82" in a["note"]
