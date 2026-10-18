import math

import pytest

from attack import tables
from attack.score import LAMBDA_FLOOR, N_POP, risk_metrics, score_record, summarize


def test_known_probabilities():
    m = risk_metrics([0.5, 0.01, 1e-3])
    assert m["bits"] == pytest.approx(1 + math.log2(100) + math.log2(1000))
    lam = (N_POP - 1) * 0.5 * 0.01 * 1e-3
    assert m["lambda"] == pytest.approx(lam)
    assert m["risk"] == pytest.approx((1 - math.exp(-lam)) / lam)


def test_lambda_to_zero_is_floored_and_risk_goes_to_one():
    m = risk_metrics([1e-6] * 5)
    assert m["lambda"] == LAMBDA_FLOOR
    assert m["risk"] == pytest.approx(1.0)


def test_large_lambda_risk_is_one_over_lambda():
    m = risk_metrics([])
    assert m["bits"] == 0
    assert m["lambda"] == pytest.approx(N_POP - 1)
    assert m["risk"] == pytest.approx(1 / (N_POP - 1))


def _a(field, value, **kw):
    return {"field": field, "value": value, "evidence": value, "rationale": "", **kw}


def test_anti_double_counting():
    attrs = [
        _a("age", "39"), _a("birth_year", "1982"), _a("absolute_dates", "2021"),
        _a("absolute_dates", "march 3rd 2021"), _a("sex", "female"),
        *[_a("other_facts", v, rarity=r) for v, r in (("collects stamps", "common"), ("owns a llama", "rare"),
                                                       ("plays the oboe", "uncommon"), ("races pigeons", "very_rare"))],
    ]
    out = score_record(attrs)
    counted = [a for a in out["attributes"] if a["counted"]]
    assert sum(a["category"] == "age" for a in counted) == 1
    birth = next(a for a in out["attributes"] if a["field"] == "birth_year")
    assert "age 39" in birth["note"]  # 2021 - 1982
    dates = [a for a in counted if a["category"] == "absolute_date"]
    assert len(dates) == 1 and dates[0]["p"] == pytest.approx(1 / 365)
    facts = sorted(a["value"] for a in counted if a["category"] == "other_facts")
    assert facts == ["owns a llama", "plays the oboe", "races pigeons"]
    expected_ps = [tables.age_range_share(39, 39), 1 / 365, 0.5, 1e-4, 1e-3, 1e-2]
    assert out["bits"] == pytest.approx(sum(-math.log2(p) for p in expected_ps))


def test_summary_aggregates():
    row = {"failed": False, "n_extracted": 4, "masking_density": 2.0,
           "dropped": [{"reason": "evidence_not_found"}, {"reason": "placeholder_value"}],
           **score_record([_a("sex", "male"), _a("state_or_region", "Washington")])}
    s = summarize([row, {"failed": True, "masking_density": 4.0}])
    assert s["n_records"] == 2 and s["n_failed"] == 1
    assert s["hallucination_rate"] == pytest.approx(1 / 4)
    assert list(s["top_leaking_attributes"]) == ["state_or_region", "sex"]
    assert s["mean_masking_density"] == pytest.approx(3.0)
