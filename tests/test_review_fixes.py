"""Regression tests for the findings of the independent review (MEDANON_PROBLEMI_CRITICI_PER_CLAUDE.txt).
All inputs are synthetic."""
import json
import subprocess
import sys

import pytest

from conftest import make_record


# ---------------------------------------------------------------- P0-02 spoken-digit identifiers
@pytest.mark.parametrize("turn,secret", [
    ("[patient] my medical record number is four five six seven eight nine .", "four five six seven eight nine"),
    ("[patient] my phone number is five five five one two three four five six seven .", "five five five one two three four five six seven"),
    ("[patient] mrn , uh , three three six five two one .", "three three six five two one"),
    ("[patient] you can call me at five five five , two three four , nine nine eight eight .", "nine nine eight eight"),
    ("[patient] it's eight one seven four four two nine nine zero one .", "eight one seven four four two nine nine zero one"),
])
def test_spoken_digit_identifiers_removed(anonymizer, turn, secret):
    out, _ = anonymizer.anonymize(make_record(turn + "\n", "Patient seen."))
    assert secret not in out["dialogue"]


@pytest.mark.parametrize("turn", [
    "[doctor] your blood pressure is one twenty over eighty .",
    "[doctor] your a1c came back at eight point three .",
    "[doctor] take it two or three times a day for five days .",
    "[doctor] rate the pain from one to ten .",
])
def test_spoken_clinical_numbers_untouched(anonymizer, turn):
    out, _ = anonymizer.anonymize(make_record(turn + "\n", "Patient seen.", eid="N1"))
    if "a1c" in turn:  # lab values may be perturbed, but only the value
        assert out["dialogue"].startswith("[doctor] your a1c came back at ")
    else:
        assert out["dialogue"] == turn + "\n"


# ---------------------------------------------------------------- P0-03 audit privacy
def test_audit_contains_no_original_surfaces_by_default(anonymizer):
    note = ("HISTORY OF PRESENT ILLNESS\n\nMr. Kowalski is a 61-year-old male. He takes Brilinta, Wixela and Flovent. "
            "Continue Lasix.\n")
    rec = make_record("[doctor] hi , mr. kowalski .\n", note)
    out, decs = anonymizer.anonymize(rec)
    blob = json.dumps(decs).lower()
    assert "kowalski" not in blob
    kept = [d for d in decs if d["action"] == "keep"]
    for d in kept:
        surface = rec[d["field"]][d["start"]:d["end"]].lower()
        assert surface not in blob, surface
        assert "text" not in d and "original" not in d and d["reason_code"]
    assert all("original" not in d and "text" not in d for d in decs)


def test_audit_includes_surfaces_only_on_request(anonymizer):
    rec = make_record("[doctor] hi , mr. kowalski .\n", "Mr. Kowalski is a 61-year-old male.")
    _, decs = anonymizer.anonymize(rec, include_text=True)
    assert any(d.get("original", "").lower().endswith("kowalski") for d in decs)


# ---------------------------------------------------------------- P1-02 perturbation bound
def test_perturbation_never_exceeds_relative_bound():
    from medanon.rules import LABS
    from medanon.transform import perturb_lab

    for lab, (_, _, (lo, hi)) in LABS.items():
        for v in (lo, (lo + hi) / 10, (lo + hi) / 4, hi / 2):
            for dec in (0, 1, 2):
                v = round(v, dec)
                for seed in range(5):
                    nv = perturb_lab(lab, v, dec, f"t{seed}", 0.03)
                    if nv is not None:
                        assert abs(nv - v) <= 0.03 * abs(v) + 1e-9, (lab, v, nv)


@pytest.mark.parametrize("lab,v,dec", [("tsh", 0.1, 1), ("tsh", 0.2, 1), ("creatinine", 0.7, 1), ("inr", 1.0, 1)])
def test_small_values_are_kept_not_distorted(lab, v, dec):
    from medanon.transform import perturb_lab

    assert all(perturb_lab(lab, v, dec, f"s{i}", 0.03) is None for i in range(20))


def test_perturbation_key_is_not_public(anonymizer):
    assert anonymizer.transformer.secret != str(anonymizer.cfg.get("seed"))


# ---------------------------------------------------------------- P1-04 / P1-03 attack metrics
def test_score_candidates_separates_outcomes():
    from medanon.evaluation.attacks import score_candidates

    res = score_candidates([[0], [0], [], [3, 1], [0, 1, 2]])
    assert res["correct_unique_rate"] == 0.2      # target 0
    assert res["wrong_unique_rate"] == 0.2        # target 1 -> only record 0
    assert res["no_candidate_rate"] == 0.2        # target 2
    assert res["target_in_candidates_rate"] == 0.4  # targets 0 and 3 (4 not in [0,1,2])
    assert abs(res["expected_reidentification"] - (1 + 0.5) / 5) < 1e-4


def test_lab_linkage_with_tolerance_beats_exact_matching():
    from medanon.evaluation.attacks import lab_linkage_attack

    recs = [{"encounter_id": f"L{i}", "note": f"Hemoglobin A1c {a}%. LDL {l}.", "dialogue": ""}
            for i, (a, l) in enumerate([(7.1, 101), (8.4, 140), (9.6, 88)])]
    outs = [{"encounter_id": r["encounter_id"], "note": r["note"].replace("7.1", "7.2").replace("140", "137").replace("9.6", "9.4"),
             "dialogue": ""} for r in recs]
    res = lab_linkage_attack(recs, outs)
    assert res["anonymised_release@tol0%"]["correct_unique_rate"] < res["anonymised_release@tol3%"]["correct_unique_rate"]


def test_inversion_attack_defeated_by_secret_key():
    from medanon.evaluation.attacks import perturbation_inversion_attack

    recs = [{"encounter_id": f"E{i}", "note": f"Hemoglobin A1c {6.6 + i / 10:.1f}%. Glucose {150 + i}.", "dialogue": ""}
            for i in range(20)]
    res = perturbation_inversion_attack(recs)
    # with a public seed the attacker recovers exact values far above chance; with the secret key
    # only coincidental unique matches remain
    assert res["exact_recovery_public_seed"] >= 0.25 and res["exact_recovery_public_seed"] > 2 * res["exact_recovery_secret_key"]


# ---------------------------------------------------------------- P1-07 people sharing a surname
def test_shared_surname_not_merged(anonymizer):
    note = ("HISTORY OF PRESENT ILLNESS\n\nJohn Smith is a 64-year-old male. His wife, Mary Smith, accompanies him. "
            "Mrs. Smith reports he has been more tired.\n")
    dlg = "[doctor] hi john .\n[patient_guest] i'm mary , his wife .\n"
    out, decs = anonymizer.anonymize(make_record(dlg, note))
    assert "John" not in out["note"] and "Mary" not in out["note"] and "Smith" not in out["note"]
    roles = {d["start"]: d.get("role") for d in decs if d["field"] == "note" and d["label"] == "PERSON"}
    john = note.index("John Smith")
    mary = note.index("Mary Smith")
    assert roles[john] == "PATIENT" and roles[mary] != "PATIENT"


def test_only_one_patient_placeholder(anonymizer):
    dlg = "[doctor] jane roe , date of birth 2/3/1950 .\n[doctor] also seeing ann poe , date of birth 4/5/1960 .\n"
    out, decs = anonymizer.anonymize(make_record(dlg, "John Doe is a 70-year-old male."))
    patient_entities = {d["replacement"] for d in decs if d.get("role") == "PATIENT"}
    assert len(patient_entities) <= 1


# ---------------------------------------------------------------- P1-08 overlap must not eat clinical words
def test_overlap_resolution_keeps_clinical_words():
    from medanon.spans import Span, resolve_overlaps

    text = "Jones renal ultrasound today"
    person = Span("note", 0, 5, "PERSON", "rule:x", 0.9, "Jones")
    org = Span("note", 0, 22, "ORGANIZATION", "spacy:ORG", 0.6, "Jones renal ultrasound")
    protected = lambda f, a, b: "ultrasound" in text[a:b]  # noqa: E731
    out = resolve_overlaps([person, org], protected)
    assert len(out) == 1 and (out[0].start, out[0].end, out[0].label) == (0, 5, "PERSON")
    assert resolve_overlaps([person, org])[0].end == 22  # without protection the union is taken


# ---------------------------------------------------------------- P2-01 long turns in the cascade
def test_long_turn_is_windowed():
    from medanon.ner import _windows

    text = "word " * 900 + "xiomara"
    wins = _windows(text, 0, len(text), 1000)
    pos = text.index("xiomara")
    assert all(e - s <= 1000 for s, e in wins) and any(s <= pos and pos + 7 <= e for s, e in wins)


# ---------------------------------------------------------------- P0-01 / P2-04 manifest and strict validation
def test_cli_writes_manifest_and_validates(tmp_path):
    from medanon.evaluation.synthetic import generate

    src = tmp_path / "in.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        for r in generate(3, seed=5):
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    dst = tmp_path / "out.jsonl"
    subprocess.run([sys.executable, "-m", "medanon.cli", "anonymize", "-i", str(src), "-o", str(dst), "--offline"],
                   check=True, capture_output=True)
    man = json.loads((tmp_path / "out.manifest.json").read_text(encoding="utf-8"))
    assert man["detectors"]["ml"]["status"] == "active" and man["detectors"]["ml"]["weights_sha256"] not in (None, "n/a")
    assert man["output"]["records"] == 3 and len(man["code_sha256"]) == 64
    assert "secret" not in json.dumps(man["config"]).lower() or man["lab_perturbation_key"].startswith("random")
    res = subprocess.run([sys.executable, "-m", "medanon.cli", "validate", "--original", str(src), "--anonymized", str(dst)],
                         capture_output=True, text=True)
    assert res.returncode == 0 and json.loads(res.stdout)["ok"]


def test_required_model_fails_loudly():
    from medanon import Anonymizer, load_config

    with pytest.raises(RuntimeError, match="prepare-models"):
        Anonymizer(load_config(overrides={"ml": {"enabled": True, "kind": "hf", "model": "nonexistent/medanon-x",
                                                 "required": True, "allow_download": False}}))


def test_qi_step_accepts_occupations_without_category(anonymizer):
    # statistical detectors produce OCCUPATION spans without a category; the QI step must treat them
    # like the transform does ("other") instead of failing when they mix with categorised rule spans
    from medanon.spans import Span

    rec = make_record("[doctor] hello .", "She works as a teacher and as a beekeeper.")
    texts, spans, reg, extra = anonymizer.detect(rec)
    spans = list(spans) + [Span("note", 0, 3, "OCCUPATION", "ml:test", 0.9, "She", {}, [])]
    assert anonymizer.quasi_identifiers((texts, spans, reg, extra), 0)
