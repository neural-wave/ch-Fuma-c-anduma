"""End-to-end run with a fake LLM: checks the plumbing and the output files, no server needed."""
import json

from attack import run

ANON = {"encounter_id": "E1", "dataset": "t", "icd10_codes": [{"code": "I10", "description": "htn"}],
        "dialogue": "[doctor] hi [PATIENT_NAME] .\n"
                    "[patient] i went to [US_STATE] to hike around mount baker . i'm an x-ray tech .",
        "note": "[PATIENT_NAME] is a 50-year-old female."}
ORIG = dict(ANON, dialogue="[doctor] hi mrs. jones .\n"
                           "[patient] i went to washington to hike around mount baker . i'm an x-ray tech .",
            note="Kate Jones is a 50-year-old female.")


def _attr(value=None, evidence=None):
    return {"value": value, "evidence": evidence, "rationale": "r" if value else None}


def fake_extract(client, text):
    out = {f: _attr() for f in ("age", "birth_year", "sex", "state_or_region", "city", "occupation",
                                  "employer_or_facility", "ethnicity", "nationality", "family_structure")}
    out |= {"absolute_dates": [], "other_facts": [],
            "age": _attr("50", "50-year-old female"), "sex": _attr("female", "50-year-old female"),
            "occupation": _attr("x-ray tech", "i'm an x-ray tech")}
    if "washington" in text:
        out["state_or_region"] = _attr("Washington", "i went to washington to hike")
    return out


def fake_recover(client, text):
    return {"guesses": [{"placeholder": "[US_STATE]", "type": "state_or_region", "guess": "Washington",
                         "reasoning": "Mount Baker is in Washington.", "confidence": "high"}]}


class FakeClient:
    model, backend, stats = "fake-model", "fake", {"llm_calls": 0, "cache_hits": 0, "retries": 0}

    def __init__(self, *a, **kw):
        pass

    def connect(self):
        pass


def test_offline_run(tmp_path, monkeypatch):
    for name, rows in (("anon.jsonl", [ANON]), ("orig.jsonl", [ORIG])):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(run, "LLMClient", FakeClient)
    monkeypatch.setattr(run, "extract", fake_extract)
    monkeypatch.setattr(run, "recover", fake_recover)
    out = tmp_path / "out"
    assert run.main(["--input", str(tmp_path / "anon.jsonl"), "--original", str(tmp_path / "orig.jsonl"),
                     "--out", str(out), "--workers", "2"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["privacy_gain"] > 0 and summary["bits_removed"] > 0 and not summary["warnings"]
    assert summary["recovery"]["overall"]["recovery_rate"] == 1.0
    row = json.loads((out / "per_record.jsonl").read_text())
    assert row["n_icd_codes"] == 1 and row["recovery"][0]["correct"]
    assert (out / "top_leaking_attributes.png").stat().st_size > 0
    assert "E1" in (out / "report.md").read_text()
