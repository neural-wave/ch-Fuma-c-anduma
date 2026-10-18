"""End-to-end behaviour on synthetic encounters (spaCy model required)."""
import json
import re
import subprocess
import sys

from conftest import make_record

DIALOGUE = ("[doctor] hi , mr. parkinson , how are you ?\n"
            "[patient] hi doctor patel . my wife , jenny , drove me in from springfield .\n"
            "[doctor] so you're a 67-year-old with parkinson's disease . your a1c was eight point three .\n"
            "[patient] i work as a truck driver at walmart . i take lasix and allegra .\n"
            "[doctor] your bell's palsy looks better . we'll see you on monday , march 3rd .\n")
NOTE = ("HISTORY OF PRESENT ILLNESS\n\nStanley Parkinson is a 67-year-old male with Parkinson's disease and Bell's palsy. "
        "He works as a truck driver. He lives in Springfield, Illinois. Takes Lasix, Allegra and metformin. "
        "Lyme disease in 2015.\n\nRESULTS\n\nHemoglobin A1c 8.3%.\n\nASSESSMENT AND PLAN\n\nFollow up with Dr. Patel.\n")
ICD = [{"code": "G20", "description": "Parkinson's disease"}, {"code": "G51.0", "description": "Bell's palsy"}]


def run(anonymizer, **kw):
    return anonymizer.anonymize(make_record(DIALOGUE, NOTE, ICD), **kw)


def test_identifiers_removed(anonymizer):
    out, _ = run(anonymizer)
    text = (out["dialogue"] + out["note"]).lower()
    # calendar years and exact ages up to 89 are published by policy (see test_policy.py)
    for leaked in ("stanley", "mr. parkinson", "patel", "jenny", "springfield", "illinois", "walmart", "3rd", "monday"):
        assert leaked not in text, leaked


def test_medical_content_preserved(anonymizer):
    out, _ = run(anonymizer)
    for f in ("dialogue", "note"):
        assert out[f].lower().count("parkinson's disease") == DIALOGUE.lower().count("parkinson's disease") if f == "dialogue" \
            else out[f].count("Parkinson's disease") == 1
    assert "Bell's palsy" in out["note"] and "bell's palsy" in out["dialogue"]
    assert "Lasix" in out["note"] and "lasix" in out["dialogue"] and "allegra" in out["dialogue"]
    assert "Lyme disease" in out["note"] and "metformin" in out["note"]


def test_consistent_role_placeholders(anonymizer):
    out, decs = run(anonymizer)
    assert "[PATIENT_NAME]" in out["note"] and "[PATIENT_NAME]" in out["dialogue"]
    assert "[CLINICIAN_NAME]" in out["note"] and "[CLINICIAN_NAME]" in out["dialogue"]
    assert "[RELATIVE_NAME]" in out["dialogue"]
    roles = {d["role"] for d in decs if d.get("role")}
    assert {"PATIENT", "CLINICIAN", "FAMILY"} <= roles


def test_lab_generalized_by_default_in_both_fields(anonymizer):
    out, _ = run(anonymizer)
    assert re.search(r"A1c \[8-9\]", out["note"]) and "8.3" not in out["note"]
    assert "eight point three" not in out["dialogue"]


def test_lab_perturbation_consistent_across_fields():
    from medanon import Anonymizer, load_config
    from medanon.lexicon import words_to_number

    anonymizer = Anonymizer(load_config("configs/standard.yaml"))  # bounded perturbation profile
    out, _ = run(anonymizer)

    note_val = float(re.search(r"A1c (\d+\.\d)%", out["note"]).group(1))
    dlg_val = words_to_number(re.search(r"a1c was ([a-z ]+?) \.", out["dialogue"]).group(1))
    assert note_val == dlg_val and note_val != 8.3 and 8.0 <= note_val < 9.0


def test_schema_icd_and_metadata_untouched(anonymizer):
    rec = make_record(DIALOGUE, NOTE, ICD)
    out, _ = anonymizer.anonymize(rec)
    assert set(out) == set(rec) and out["icd10_codes"] == ICD and out["encounter_id"] == rec["encounter_id"]
    assert out["dataset"] == rec["dataset"]


def test_every_decision_explained_and_deterministic(anonymizer):
    out1, d1 = run(anonymizer)
    out2, d2 = run(anonymizer)
    assert out1 == out2 and d1 == d2
    assert d1 and all(d["explanation"] for d in d1)
    assert all("original" not in d for d in d1)  # audit does not leak originals by default


def test_no_icd_mode_still_protects_eponyms():
    from medanon import Anonymizer, load_config

    a = Anonymizer(load_config(overrides={"use_icd_context": False}))
    out, _ = a.anonymize(make_record(DIALOGUE, NOTE, []))
    assert "Parkinson's disease" in out["note"] and "Bell's palsy" in out["note"]
    assert "Stanley" not in out["note"]


def test_line_structure_is_preserved(anonymizer):
    note = "PLAN\n\nNext visit March\n2025 with Dr. Patel.\nCall 555-123-4567.\n"
    out, _ = anonymizer.anonymize(make_record("[doctor] see you in march\n[patient] okay .\n", note))
    assert out["note"].count("\n") == note.count("\n") and "Patel" not in out["note"] and "March" not in out["note"]
    assert out["dialogue"].count("\n") == 2


def test_missing_ml_model_degrades_gracefully():
    from medanon import Anonymizer, load_config

    a = Anonymizer(load_config(overrides={"ml": {"enabled": True, "kind": "hf", "model": "nonexistent/medanon-test-model",
                                                 "required": False}}))
    assert a.ml is None and a.ml_status.startswith("unavailable")
    out, _ = a.anonymize(make_record(DIALOGUE, NOTE, ICD))
    assert "Stanley" not in out["note"]


def test_placeholder_format_option():
    from medanon import Anonymizer, load_config

    a = Anonymizer(load_config(overrides={"placeholder_format": "<{}>"}))
    out, _ = a.anonymize(make_record(DIALOGUE, NOTE, ICD))
    assert "<PATIENT_NAME>" in out["note"] and "[PATIENT_NAME]" not in out["note"]
    assert out["dialogue"].startswith("[doctor]")  # speaker tags untouched


def test_k_anonymity_never_widens_exact_ages(anonymizer):
    # default policy: exact ages are published on purpose; the QI step may only act on other QIs and
    # must report honestly that the records stay unique
    recs = [make_record("[doctor] hello .", f"She is a {age}-year-old female who works as a teacher.", eid=f"K{i}")
            for i, age in enumerate([31, 33, 35, 81])]
    results, stats = anonymizer.anonymize_dataset(recs)
    notes = [r["note"] for r, _ in results]
    assert all(f"{a}-year-old" in n for a, n in zip([31, 33, 35, 81], notes))
    v = stats["qi_verification"]
    assert v["records_below_k"] == 4 and v["qi_set"].startswith("exact age")


def test_k_anonymity_escalates_unique_records():
    from medanon import Anonymizer, load_config

    anonymizer = Anonymizer(load_config(overrides={"policies": {"AGE": {"action": "generalize"}}, "extra_detectors": []}))
    recs = [make_record("[doctor] hello .", f"She is a {age}-year-old female who works as a teacher.", eid=f"K{i}")
            for i, age in enumerate([31, 33, 35, 81])]
    results, stats = anonymizer.anonymize_dataset(recs)
    notes = [r["note"] for r, _ in results]
    assert all("[30-39]" in n for n in notes[:3])
    assert "[AGE]" in notes[3] and "81" not in notes[3]  # escalated up to suppression
    assert "[OCCUPATION]" in notes[3] and "[OCCUPATION: education]" in notes[0]
    assert stats["escalation_levels"].get("0") == 3
    # the post-hoc check is honest: the record stays distinguishable (by its escalation pattern)
    v = stats["qi_verification"]
    assert v["k"] == 3 and v["records_below_k"] == 1 and v["min_class_size"] == 1


def test_synthetic_benchmark_regression(anonymizer):
    from medanon.evaluation.synthetic import generate, score

    recs = generate(25, seed=99)
    outs, decs = zip(*(anonymizer.anonymize({k: v for k, v in r.items() if not k.startswith("_")}) for r in recs))
    res = score(recs, list(outs), list(decs))
    # raw coverage keeps its historical definition; the deliberate retention of exact ages and calendar
    # years (policy 2026-10-17) is accounted for explicitly, never hidden from the denominator
    assert res["policy_compliant_recall_coverage"] >= 0.95 and res["trap_preservation"] >= 0.99 and res["spurious_replacements"] <= 2
    assert set(res["kept_by_policy"]) <= {"AGE", "YEAR"}
    assert res["recall_coverage"] < res["policy_compliant_recall_coverage"]


def test_cli_roundtrip(tmp_path):
    from medanon.evaluation.synthetic import generate

    src = tmp_path / "in.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        for r in generate(3, seed=1):
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    dst = tmp_path / "out.jsonl"
    subprocess.run([sys.executable, "-m", "medanon.cli", "anonymize", "-i", str(src), "-o", str(dst)], check=True,
                   capture_output=True)
    rows = [json.loads(line) for line in open(dst, encoding="utf-8")]
    assert len(rows) == 3 and all(r["icd10_codes"] for r in rows)
    assert (tmp_path / "out.audit.jsonl").exists()
