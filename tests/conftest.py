import pytest


@pytest.fixture(scope="session")
def anonymizer():
    from pathlib import Path

    from medanon import Anonymizer, load_config

    # the historical suite pins the legacy exact-age policy it was written for; the release default (age bands)
    # is covered by tests/test_hardening.py and tests/test_release_default.py
    return Anonymizer(load_config(Path(__file__).resolve().parents[1] / "configs" / "exact_age.yaml"))


def make_record(dialogue: str, note: str, icd=None, eid: str = "T001") -> dict:
    return {"encounter_id": eid, "dataset": "aci", "dialogue": dialogue, "note": note,
            "icd10_codes": icd or [{"code": "I10", "description": "Essential (primary) hypertension"}]}
