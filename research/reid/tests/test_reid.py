"""Hand-computable fixtures for the re-identification benchmark (run: pytest research/reid/tests -q)."""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reid.attack.compare import active, deterministic, levels_numpy  # noqa: E402
from reid.attack.extract import Vocab, extract, register_features  # noqa: E402
from reid.evaluate import metrics, pool_posterior  # noqa: E402

COLS = ("nm_found", "nm", "sex", "by_lo", "by_hi", "dob_md", "city", "occ_title", "occ_cat", "marital", "kids_lo",
        "kids_hi", "work", "hb", "race", "ht_lo", "ht_hi", "wt_lo", "wt_hi", "bmi_lo", "bmi_hi", "ev", "icd", "vis_y", "vis_ym")


def frame(rows):
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = None
    return df


# ------------------------------------------------------------------ comparison levels
R = frame([
    {"nm_found": ["kowalczyk"], "sex": "F", "by_lo": 1975, "by_hi": 1985, "occ_cat": "transport", "kids_lo": 3, "kids_hi": 20,
     "ht_lo": 170, "ht_hi": 175, "ev": [2015, 2019], "vis_y": [2025]},
    {"sex": None, "by_lo": 1978, "by_hi": 1978, "occ_title": "truck driver", "occ_cat": "transport", "kids_lo": 2, "kids_hi": 2},
])
A = frame([
    {"nm": ["anna", "kowalczyk"], "sex": "F", "by_lo": 1978, "by_hi": 1978, "occ_title": "truck driver", "occ_cat": "transport",
     "kids_lo": 4, "kids_hi": 4, "ht_lo": 166, "ht_hi": 182, "ev": [2015, 2019], "vis_y": [2025], "vis_ym": ["2025-03"]},
    {"nm": ["bea", "novak"], "sex": "M", "by_lo": 1990, "by_hi": 1990, "occ_title": "nurse", "occ_cat": "healthcare",
     "kids_lo": 2, "kids_hi": 2, "ht_lo": 150, "ht_hi": 160, "ev": [2001], "vis_y": [2024], "vis_ym": ["2024-01"]},
    {"nm": ["kowalczyk", "smith"], "sex": None, "by_lo": 1973, "by_hi": 1979, "occ_title": None, "occ_cat": None},
])


def test_levels_by_hand():
    L = levels_numpy(R, A, active(R, A))
    # names: rec0 found {kowalczyk}: row0 last name -> 2, row1 none -> 0, row2 first name kowalczyk -> 1; rec1 no names -> null
    assert L["names"].tolist() == [[2, 0, 1], [-1, -1, -1]]
    assert L["sex"].tolist() == [[1, 0, -1], [-1, -1, -1]]
    # birth year: rec0 [1975,1985] (w=10): row0 overlap wide -> 1; row1 disjoint -> 0; row2 [1973,1979] overlap, w=10 -> 1
    # rec1 [1978,1978]: row0 exact -> 4; row1 0; row2 w=6 -> 2
    assert L["by"].tolist() == [[1, 0, 1], [4, 0, 2]]
    # occupation: rec0 category only -> row0 category 1, row1 different 0, row2 null; rec1 title -> row0 title 2, row1 0
    assert L["occ"].tolist() == [[1, 0, -1], [2, 0, -1]]
    # children: rec0 [3+] -> row0 (4) compatible 1, row1 (2) 0; rec1 exact 2 -> row0 0, row1 exact 2
    assert L["kids"].tolist() == [[1, 0, -1], [0, 2, -1]]
    assert L["ht"].tolist() == [[1, 0, -1], [-1, -1, -1]]
    assert L["ev"].tolist() == [[2, 0, -1], [-1, -1, -1]]
    assert L["visit"].tolist() == [[1, 0, -1], [-1, -1, -1]]
    compat, n_agree = deterministic(L)
    assert compat.tolist() == [[True, False, True], [False, False, True]]
    assert n_agree[0].tolist() == [8, 0, 2] and n_agree[1].tolist() == [2, 1, 1]


def test_splink_gammas_equal_numpy():
    pytest.importorskip("splink")
    from reid.attack.run import splink_scores
    big_a = pd.concat([A] * 40, ignore_index=True)  # enough rows for u sampling
    big_a["key"] = [f"k{i}" for i in range(len(big_a))]
    attrs = active(R, big_a)
    L = levels_numpy(R, big_a, attrs)
    _, gam, _ = splink_scores(R, big_a, attrs, seed=1)
    for a in attrs:
        assert (gam[a] == L[a]).all(), a


# ------------------------------------------------------------------ pool posterior and metrics
def test_pool_posterior_by_hand():
    s = np.log2([8.0, 1.0, 1.0])
    assert pool_posterior(s, 3, 1.0)[0] == pytest.approx(0.8)          # 8 / (8+1+1)
    assert pool_posterior(s, 3, 0.5)[0] == pytest.approx(8 / 13)       # 8 / (1*3 + 10)
    assert pool_posterior(s + 50, 3, 0.5)[0] == pytest.approx(0.8, abs=1e-9)  # huge evidence: absence option vanishes


def test_metrics_by_hand():
    df = pd.DataFrame({"pid": ["p1", "p1", "p2", "p3"], "present": [True, True, True, True], "hard": [False] * 4,
                       "fs_rank": [1, 3, 1, 10], "fs_id": [True, False, True, True], "fs_correct": [True, False, False, False]})
    m = metrics(df, "fs", 100, "closed")
    assert m["top1"][0] == 0.5 and m["coverage"][0] == 0.75 and m["precision"][0] == pytest.approx(1 / 3, abs=1e-4)
    assert m["success_over_eligible"][0] == 0.25 and m["false_identification_rate"][0] == 0.5
    assert m["mrr"][0] == pytest.approx((1 + 1 / 3 + 1 + 0.1) / 4, abs=1e-4)
    assert m["patient_reidentified"][0] == pytest.approx(1 / 3, abs=1e-4)        # only p1 has a correct identification
    assert m["random_top1"] == 0.01


# ------------------------------------------------------------------ parsing of published (anonymised) formats
@pytest.fixture(scope="module")
def voc():
    reg = pd.DataFrame({"key": ["k1"], "first": ["anna"], "last": ["kowalczyk"], "city": ["Springfield"],
                        "occupation": ["truck driver"], "hobby": ["salsa dancing"]})
    return Vocab(reg, {"driver": "transport"}, set())


def rec(note, dia="", icd=("I10",)):
    return {"encounter_id": "x", "note": note, "dialogue": dia, "icd10_codes": [{"code": c} for c in icd]}


def test_band_and_visit_year(voc):
    f = extract(rec("Date of Service: [DATE] 2025\n\nHPI\n[PATIENT_NAME] is a [40-49]-year-old female."), voc)
    assert (f["by_lo"], f["by_hi"]) == (2025 - 49 - 1, 2025 - 40) and f["vis_y"] == [2025] and f["sex"] == "F"
    assert f["ev"] is None  # the visit-header year is not a clinical event year


def test_exact_age_without_visit_year_uses_window(voc):
    f = extract(rec("The patient is a 47-year-old male."), voc)
    assert (f["by_lo"], f["by_hi"]) == (2023 - 47 - 1, 2026 - 47)


def test_published_birth_year_and_calendar_year_not_birth(voc):
    f = extract(rec("[DOB] 1978\nShe is a 47 year old woman.\n- Type 2 diabetes (diagnosed 1978)"), voc)
    assert (f["by_lo"], f["by_hi"]) == (1978, 1978) and f["ev"] == [1978]
    g = extract(rec("[DOB]\nShe is a [40-49] year old woman. Hypertension since 1978."), voc)
    assert g["by_lo"] != 1978 or g["by_hi"] != 1978  # a retained calendar year is never read as the birth year
    assert g["ev"] == [1978]


def test_top_coded_age(voc):
    f = extract(rec("Mr. [PATIENT_NAME] is a 90+-year-old male."), voc)
    assert f["by_hi"] == 2026 - 90 and f["by_lo"] < 1930


def test_body_measurements_and_categories(voc):
    f = extract(rec("VITALS\nHeight [170-175] cm, Weight 181 lbs, BMI [25-29.9].\nSOCIAL HISTORY\nWorks as a "
                    "[OCCUPATION: transport]. Married with [3+] children."), voc)
    assert (f["ht_lo"], f["ht_hi"]) == (170.0, 175.0)
    assert f["wt_lo"] == pytest.approx(180.5 / 2.2046, abs=0.01) and f["wt_hi"] == pytest.approx(181.5 / 2.2046, abs=0.01)
    assert (f["bmi_lo"], f["bmi_hi"]) == (25.0, 30.0)
    assert f["occ_cat"] == "transport" and f["occ_title"] is None
    assert (f["kids_lo"], f["kids_hi"]) == (3, 20) and f["marital"] == "M"


def test_residual_dictionary_hits(voc):
    f = extract(rec("Ms. Kowalczyk works as a truck driver.", "[patient] i live over in springfield . salsa dancing mostly ."), voc)
    assert f["nm_found"] == ["kowalczyk"] and f["occ_title"] == "truck driver" and f["occ_cat"] == "transport"
    assert f["city"] == "springfield" and f["hb"] == ["salsa dancing"]
    g = extract(rec("[PATIENT_NAME] lives in [CITY]."), voc)
    assert g["nm_found"] is None and g["city"] is None


def test_register_features_intervals():
    reg = pd.DataFrame({"key": ["a", "b"], "first": ["x", "y"], "last": ["p", "q"], "birth_date": ["1980-05-02", None],
                        "age_approx": [None, 40], "age_tol": [None, 2], "height_lo": [170, None], "height_hi": [180, None],
                        "weight_lo": [70, None], "weight_hi": [80, None]})
    A2 = register_features(reg, {})
    assert (A2.loc[0, "by_lo"], A2.loc[0, "by_hi"], A2.loc[0, "dob_md"]) == (1980, 1980, "05-02")
    assert (A2.loc[1, "by_lo"], A2.loc[1, "by_hi"]) == (2026 - 40 - 2 - 1, 2026 - 40 + 2)
    assert A2.loc[0, "bmi_lo"] == pytest.approx(70 / 1.8 ** 2, abs=0.01) and A2.loc[0, "bmi_hi"] == pytest.approx(80 / 1.7 ** 2, abs=0.01)


# ------------------------------------------------------------------ isolation of the attacker
def test_attacker_imports_no_generator_or_truth_code():
    banned = {"world", "render", "vocab", "blocks", "evaluate", "clinical", "bench"}
    for f in (ROOT / "reid" / "attack").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")
                assert not (set(mod) & banned) and not (node.level >= 2), (f.name, node.module)
            elif isinstance(node, ast.Import):
                assert not any(set(a.name.split(".")) & banned for a in node.names), f.name


def test_attacker_refuses_private_inputs(tmp_path):
    from reid.attack.run import _guard
    with pytest.raises(SystemExit):
        _guard(str(tmp_path / "private" / "keys_A.csv"))
    with pytest.raises(SystemExit):
        _guard(str(tmp_path / "release_input" / "encounters.jsonl"))
    assert _guard(str(tmp_path / "anonymized" / "x.jsonl"))


def test_spoken_age_answer_and_two_digit_dob(voc):
    f = extract(rec("The patient is a [40-49]-year-old male.", "[doctor] you're how old now ?\n[patient] uh , 49"), voc)
    assert (f["by_lo"], f["by_hi"]) == (2023 - 49 - 1, 2026 - 49)  # exact dialogue answer refines the note band
    g = extract(rec("The patient presents.", "[doctor] you're how old now ?\n[patient] uh , 49"), voc)
    assert (g["by_lo"], g["by_hi"]) == (2023 - 49 - 1, 2026 - 49)
    h = extract(rec("The patient presents.", "[doctor] can i get your date of birth ?\n[patient] [DOB] 78"), voc)
    assert (h["by_lo"], h["by_hi"]) == (1978, 1978)
    k = extract(rec("The patient presents.", "[doctor] what's your date of birth ?\n[patient] okay , [DATE] 05"), voc)
    assert (k["by_lo"], k["by_hi"]) == (2005, 2005)


def test_open_world_false_match_by_hand():
    df = pd.DataFrame({"pid": ["p1", "p2", "p3", "p4"], "present": [True, True, False, False], "hard": [False] * 4,
                       "fs_rank": [1, 2, None, None], "fs_id": [True, True, True, False],
                       "fs_correct": [True, False, False, False]})
    m = metrics(df, "fs", 100, "open")
    assert m["false_match_rate_absent"][0] == 0.5          # 1 of 2 absent targets wrongly attributed
    assert m["success_over_eligible"][0] == 0.5            # 1 of 2 present targets correctly identified
    assert m["false_identification_rate"][0] == 0.5        # p2 (wrong person) + p3 (absent) over 4 queries
