"""Builds VULNERABILITIES.json from the measured result files (so every number is traceable):
results/policy_audit_train140.json, results/llm_attack_runs.json, synthetic_regressions/outputs/*.check.json.

    .venv/bin/python attack/handoff/make_vulnerabilities.py
"""
import json
from pathlib import Path

H = Path(__file__).parent
AUDIT = json.loads((H / "results" / "policy_audit_train140.json").read_text())
LLM = json.loads((H / "results" / "llm_attack_runs.json").read_text())
POLICIES = ("default", "age_bands", "age_bands_strict_dates")
CHECKS = {p: json.loads((H / "synthetic_regressions" / "outputs" / f"{p}.check.json").read_text()) for p in POLICIES}


def probe(eid: str) -> dict:
    """Pass/fail of every assertion of one synthetic probe under each policy (actual ceff209 output)."""
    out = {}
    for p in POLICIES:
        rs = [r for r in CHECKS[p] if r["encounter_id"] == eid]
        out[p] = {f"{r['kind']}:{r['expect']}:/{r['regex']}/": "pass" if r["pass"] else "FAIL" for r in rs}
    return out


def anon_text(eid: str, policy: str = "age_bands") -> str:
    for line in (H / "synthetic_regressions" / "outputs" / f"{policy}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["encounter_id"] == eid:
            return r["dialogue"] + "\n---NOTE---\n" + r["note"]
    return ""


def freq(metric: str) -> dict:
    return {p: f"{AUDIT[p][metric]['records']}/{AUDIT[p][metric]['of']}" for p in POLICIES}


SCOPE = ("MedAnon ceff209, full detectors (rules+spaCy/Presidio+BERT+OpenMed), --offline. 'train140' = the 140 "
         "Bosch training encounters; synthetic probes = attack/handoff/synthetic_regressions (invented text).")

V = [
    {
        "id": "V1", "title": "Adult birth year survives age_bands when no adult age is stated in the record",
        "category": "quasi-identifier residue (birth year)", "classification": "incorrect transformation",
        "severity": "high per instance; prevalence 0/140 observed under age_bands", "confidence": "high (mechanism)",
        "auxiliary_information": "Visit year (encounter year is public for this dataset or inferable) and a population "
                                 "table of birth cohorts. No auxiliary identity database needed.",
        "mechanism": "A birth year is a one-year birth cohort (~6.3 bits, the same as an exact age). With the visit year "
                     "it gives the exact age back, which is exactly what age_bands exists to prevent. configs/age_bands.yaml "
                     "says the adult birth year is removed; it is only removed when an adult AGE is also detected in the record.",
        "frequency": {"train140_visible_birth_year_4digit": freq("birth_year_4digit_visible"),
                      "train140_visible_birth_year_2digit": freq("birth_year_2digit_visible"),
                      "note": "Records whose anonymized text shows a 4-/2-digit year next to DOB/born/[DOB]. Under age_bands "
                              "all 4 remaining hits (D2N099, D2N100, D2N183, D2N185) are children (kept by design): adult "
                              "birth-year residue is 0/140. In these transcripts the age is always stated next to the DOB, "
                              "so the bug only triggers on records without a stated age."},
        "synthetic_repro": {"probe": "SYN-DOB-01 (no age stated) vs SYN-DOB-02 (52-year-old stated)",
                            "results": {"SYN-DOB-01": probe("SYN-DOB-01"), "SYN-DOB-02": probe("SYN-DOB-02")},
                            "actual_output_age_bands": {"SYN-DOB-01": anon_text("SYN-DOB-01"), "SYN-DOB-02": anon_text("SYN-DOB-02")}},
        "expected_behavior": "Under age_bands, '[DOB]' with no year whenever reference_year - birth_year >= 18, independent of "
                             "whether an age was detected; two-digit years ('71', 03/03/71) resolved to 19xx/20xx first. "
                             "Paediatric birth years (age < 18) kept.",
        "likely_component": "src/medanon/transform.py (DOB policy: adult decision coupled to detected AGE); DOB 2-digit year parsing",
        "clinical_consequence_of_fix": "None for adults (the age band carries the clinical information). Keep paediatric years.",
        "regression_tests": ["SYN-DOB-01 privacy assertions must pass under age_bands", "SYN-DOB-02 stays passing",
                             "a child ('date of birth june 2nd 2019') keeps its year under age_bands",
                             "durations/clinical years ('surgery in 2018', '20 pack-years') unchanged"],
        "recommendation": "fix before submission if cheap (it silently defeats age_bands); rank after V3 and V2",
    },
    {
        "id": "V2", "title": "Clinical words masked as identifiers (over-redaction)",
        "category": "clinical integrity", "classification": "incorrect transformation (false-positive detection)",
        "severity": "medium (clinical), none (privacy)", "confidence": "high",
        "auxiliary_information": "n/a (utility loss, not a privacy leak)",
        "mechanism": "NER labels exam/anatomy words, symptoms, OTC products and order words as PERSON/LOCATION/EMPLOYER/"
                     "ORGANIZATION/OCCUPATION: 'lungs'->[PERSON], 'neurovascular'->[LOCATION], 'barky cough'->[EMPLOYER], "
                     "'Biofreeze'/'referral'->[ORGANIZATION], 'heart doctor'/'kidney doctors'->[OCCUPATION]. The masked words are "
                     "not identifying, so masking them only loses clinical content (exam findings, croup symptom).",
        "frequency": {"train140_records_with_overredaction": freq("records_with_overredaction"),
                      "train140_blocks_by_category_default": AUDIT["default"]["replacement_blocks_by_category"],
                      "terms_default_policy": {"lungs->[PERSON*]": "5 records", "neurovascular->[LOCATION]": "3 records",
                                               "barky cough->[EMPLOYER]": "1 record (3x)", "Biofreeze->[ORGANIZATION]": "1",
                                               "referral->[ORGANIZATION]": "1", "specialty->[OCCUPATION]": "2"}},
        "synthetic_repro": {"probes": "SYN-OVER-01/03 reproduce (Biofreeze in 'use some X on the neck'; 'neurovascular's intact'; "
                                      "'Neurovascular status intact'); SYN-OVER-02/04 do NOT reproduce (lungs, barky cough, referral "
                                      "are context-dependent: seen only in real transcripts)",
                            "results": {e: probe(e) for e in ("SYN-OVER-01", "SYN-OVER-03", "SYN-OVER-04")},
                            "actual_output_age_bands": {"SYN-OVER-03": anon_text("SYN-OVER-03")}},
        "expected_behavior": "Clinical vocabulary is never a PERSON/LOCATION/ORGANIZATION/EMPLOYER/OCCUPATION span.",
        "likely_component": "post-NER filter (medical.py / lexicon guard) applied to ML-detector spans (BERT/OpenMed label maps)",
        "clinical_consequence_of_fix": "Positive: restores exam findings and symptoms. Risk: a real name that is also an "
                                       "anatomy word; keep names in name position ('okay frank , your lungs...').",
        "regression_tests": ["SYN-OVER-01..04 clinical assertions all pass", "SYN-OVER-02: 'frank' still masked",
                             "attack.audit_masks on a labelled set: clinical_term blocks -> 0"],
        "recommendation": "fix before submission (protects Bosch medical-concept recall)",
    },
    {
        "id": "V3", "title": "Exact adult age retained under the default policy",
        "category": "quasi-identifier residue (age)", "classification": "intended policy retention (default)",
        "severity": "high under default, resolved by age_bands", "confidence": "high",
        "auxiliary_information": "Population age distribution (census).",
        "mechanism": "Exact age = one-year cohort, ~6.3 bits: the largest single quasi-identifier in all 8 records attacked "
                     "with the LLM (both runs). A 10-year band costs ~3 bits.",
        "frequency": {"train140_exact_adult_age_visible": freq("exact_adult_age_visible"),
                      "train140_mean_max_age_bits": {p: round(AUDIT[p]["mean_age_bits_max"], 2) for p in POLICIES}},
        "synthetic_repro": {"probe": "SYN-AGE-01", "results": probe("SYN-AGE-01")},
        "expected_behavior": "Use configs/age_bands.yaml (already the recommended configuration).",
        "likely_component": "configuration choice",
        "clinical_consequence_of_fix": "Decade bands lose age-specific thresholds (screening 45 vs 50, dosing); paediatric ages exact.",
        "regression_tests": ["SYN-AGE-01 under age_bands", "'smoked for 20 years' and '20 pack-year' unchanged"],
        "recommendation": "ship age_bands.yaml (then V1 becomes the main age leak)",
    },
    {
        "id": "V4", "title": "Named landmark next to a masked place gives the place back",
        "category": "contextual linkage / attribute recovery", "classification": "detection miss",
        "severity": "medium per instance, rare", "confidence": "high (mechanism), low (prevalence)",
        "auxiliary_information": "General world knowledge (an LLM): 'mount baker' -> Washington.",
        "mechanism": "The state and a lodge are masked but 'Mount Hood' / 'Mount Baker' stays; the LLM recovered the masked "
                     "state with high confidence (eval D2N001, older MedAnon build).",
        "frequency": {"llm_recovery": "1 of 1 recoverable masked place recovered (eval D2N001, old build); 1/7 scorable guesses overall",
                      "train140_landmark_within_120_chars_of_masked_place": "0/140 (default; heuristic scan, one false positive excluded)"},
        "synthetic_repro": {"probe": "SYN-LANDMARK-01", "results": probe("SYN-LANDMARK-01"),
                            "actual_output_age_bands": anon_text("SYN-LANDMARK-01")},
        "expected_behavior": "Named natural landmarks (Mount/Mt X, Lake X, X National Park, X River) masked as [LOCATION], "
                             "at least in sentences that already contain a masked place.",
        "likely_component": "LOCATION detection (spaCy FAC/LOC labels or a landmark rule)",
        "clinical_consequence_of_fix": "None expected; keep eponyms and scores ('Lake Louise score') as negative tests.",
        "regression_tests": ["SYN-LANDMARK-01 privacy assertions", "negative: 'mount the brace', 'lake louise score' unchanged"],
        "recommendation": "fix if time allows (low prevalence); otherwise document",
    },
    {
        "id": "V5", "title": "Bare numeric answer to 'how old are you?' kept",
        "category": "quasi-identifier residue (age)", "classification": "detection miss",
        "severity": "high per instance, rare", "confidence": "high (synthetic), prevalence low",
        "auxiliary_information": "None beyond the age table.",
        "mechanism": "Lead L1 from the other attacker. '[doctor] how old are you now ? [patient] 54 .' keeps 54 under every policy.",
        "frequency": {"train140_bare_age_answer_after_patient_age_question": freq("bare_age_answer_after_age_question")},
        "synthetic_repro": {"probe": "SYN-AGE-02", "results": probe("SYN-AGE-02")},
        "expected_behavior": "A bare number in the reply to an age question about the patient is treated as AGE.",
        "likely_component": "rules: dialogue-turn context for AGE",
        "clinical_consequence_of_fix": "None (questions about relatives' ages must not trigger: 'how old was your mother' negative test).",
        "regression_tests": ["SYN-AGE-02", "negative: 'how old was your mother ? mom was 40' is about a relative (policy choice)"],
        "recommendation": "cheap; fix if V1/V2 are done",
    },
]
NOT_REPRODUCED = [
    {"id": "L3", "lead": "Blood pressure values replaced with [ORGANIZATION]",
     "result": "not reproduced at ceff209: 0 BP masks in 9 synthetic forms (SYN-BP-01/02, 3 policies) and 0 measurement_bp "
               "replacement blocks in train140 under any policy",
     "results": {e: probe(e) for e in ("SYN-BP-01", "SYN-BP-02")}},
    {"id": "L4", "lead": "Over-redaction of clinically important values and dates",
     "result": "partly intended policy: LAB_VALUE generalize turns 'ejection fraction 45%' into '[40-50]%' and 'A1c 8.2' into "
               "'[8-9]' (SYN-KEEP-01). Medications, doses and relative chronology are kept. My attacker never prices lab "
               "values, so it has no evidence for or against lab generalization; note EF [40-50] straddles the HFrEF (<=40) / "
               "HFmrEF boundary.",
     "results": {"SYN-KEEP-01": probe("SYN-KEEP-01")}},
    {"id": "A1", "lead": "Reproducibility: manifest code_sha256 differs between machines for the same commit",
     "result": "manifest.code_hash() hashes src/medanon/env/ (a git-ignored virtualenv inside the package). Same ceff209 "
               "source hashes to 4ff71990... here vs ac834ee0 elsewhere. Exclude env/ (or hash git-tracked files only).",
     "classification": "evaluation artifact", "severity": "low", "recommendation": "trivial fix before submission"},
]
POLICY_TABLE = {p: {k: (f"{v['records']}/{v['of']}" if isinstance(v, dict) and "records" in v else v)
                    for k, v in AUDIT[p].items() if k not in ("replacement_blocks_by_category", "anonymized", "original")}
                for p in POLICIES}
SYNTH_TABLE = {p: {"privacy_pass": f"{sum(r['pass'] for r in CHECKS[p] if r['kind'] == 'privacy')}/"
                                   f"{sum(r['kind'] == 'privacy' for r in CHECKS[p])}",
                   "clinical_pass": f"{sum(r['pass'] for r in CHECKS[p] if r['kind'] == 'clinical')}/"
                                    f"{sum(r['kind'] == 'clinical' for r in CHECKS[p])}"} for p in POLICIES}

doc = {"scope": SCOPE, "medanon_commit": "ceff209", "severity_scale": "high = ~6 bits (one-year cohort) or more per record; "
       "medium = a few bits or clinical content lost; low = provenance/reporting",
       "not_measured": ["Top-1/Top-5 linkage, linkage success, coverage, precision, wrong attributions and absent-target false "
                        "matches: this attacker estimates population uniqueness and recovers masked values; it has no "
                        "auxiliary identity database or synthetic population. Use the Splink attacker for those."],
       "vulnerabilities": V, "leads_and_artifacts": NOT_REPRODUCED,
       "policy_comparison_train140": POLICY_TABLE, "synthetic_probe_pass_rates": SYNTH_TABLE,
       "llm_attack_runs": LLM["runs"]}
(H / "VULNERABILITIES.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"policy": POLICY_TABLE, "synthetic": SYNTH_TABLE}, indent=1))
