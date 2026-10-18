"""Configuration: built-in defaults deep-merged with an optional YAML file."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "fields": ["dialogue", "note"],          # only these text fields are modified
    "use_icd_context": True,                 # ICD-10 descriptions as medical evidence (bonus: set False)
    "role_aware_placeholders": True,         # [PATIENT_NAME] / [CLINICIAN_NAME] / [RELATIVE_NAME] vs [PERSON]
    "seed": 13,                              # reproducible perturbations
    "placeholder_format": "[{}]",            # e.g. "<{}>" if square brackets clash with a downstream parser
    "k_anonymity": {"enabled": True, "k": 3},  # dataset-level escalation for unique QI combinations
    "spacy": {"enabled": True, "model": "en_core_web_lg", "presidio_patterns": True, "truecase_dialogue": True},
    # transformer NER for lowercase dialogue, run only on turns with unexplained name-like tokens
    # (cascade). required=True: if the model cannot be loaded the run stops instead of silently using
    # a different pipeline; allow_download=True permits a one-time weight download (setup only).
    "ml": {"enabled": True, "kind": "hf", "model": "dslim/bert-base-NER-uncased", "threshold": 0.5,
           "mode": "cascade", "fields": ["dialogue"], "required": True, "allow_download": True},
    # default = maximum-privacy profile: OpenMed PII SuperClinical over both fields (held-out adversarial
    # recall 0.786 -> 0.886, ~12x CPU time). configs/standard.yaml switches it off.
    "extra_detectors": [{"kind": "hf", "model": "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1",
                         "label_map": "openmed-pii", "mode": "full", "threshold": 0.5,
                         "fields": ["dialogue", "note"], "allow_download": True}],
    "min_score": {"PERSON": 0.5, "LOCATION": 0.5, "ORGANIZATION": 0.6, "NATIONALITY": 0.5, "OCCUPATION": 0.5},
    "policies": {
        "PERSON": {"action": "placeholder"},
        # release policy (= configs/age_bands.yaml): exact paediatric age 0-17, adult bands [18-19], [20-29] ... [80-89],
        # "90+"; adult birth years removed. configs/exact_age.yaml restores the previous exact-age-0-89 default.
        "AGE": {"action": "band", "exact_below": 18, "top_code": 90, "band_width": 10, "format": "[{band}]"},
        # day and month removed, the calendar year kept ("[DOB] 1985"); never for patients aged 90+
        "DOB": {"action": "generalize", "placeholder": "[DOB]", "keep_year": True, "reference_year": 2026},
        "DATE": {"action": "generalize", "placeholder": "[DATE]", "keep_year": True},
        "YEAR": {"action": "keep", "placeholder": "[YEAR]"},  # clinical chronology ("surgery in 2018")
        "MONTH": {"action": "placeholder", "placeholder": "[MONTH]"},
        "WEEKDAY": {"action": "placeholder", "placeholder": "[WEEKDAY]"},
        "HOLIDAY": {"action": "placeholder", "placeholder": "[HOLIDAY]"},
        "ID": {"action": "placeholder", "placeholder": "[ID]"},
        "CONTACT": {"action": "placeholder", "placeholder": "[CONTACT]"},
        "ADDRESS": {"action": "placeholder", "placeholder": "[ADDRESS]"},
        "LOCATION": {"action": "placeholder", "placeholder": "[LOCATION]"},
        "ORGANIZATION": {"action": "placeholder", "placeholder": "[ORGANIZATION]"},
        "OCCUPATION": {"action": "generalize"},
        "NATIONALITY": {"action": "placeholder", "placeholder": "[NATIONALITY]"},
        "ETHNICITY": {"action": "keep", "placeholder": "[ETHNICITY]"},  # clinically relevant; suppressed only by k-anonymity
        "LAB_VALUE": {"action": "generalize", "max_relative": 0.03},
        # body height/weight -> 5 cm / 2 in / 5 kg / 10 lb intervals, BMI -> WHO class; exact where clinically needed
        "ANTHROPOMETRIC": {"action": "generalize", "cm_width": 5, "in_width": 2, "kg_width": 5, "lb_width": 10},
        "FAMILY_SIZE": {"action": "generalize", "exact_up_to": 2},  # "four children" -> "[3+] children"  # category interval; "perturb" = bounded noise
        "RELATIVE_DATE": {"action": "keep", "placeholder": "[RELATIVE_DATE]"},  # durations/seasons drive the clinical timeline
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _merge(cfg, overrides or {})
