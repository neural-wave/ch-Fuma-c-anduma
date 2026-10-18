import pytest

from attack.data import PLACEHOLDER_RE
from attack.verify import clean_value, verify

TEXT = ("[doctor] what do you do for a living ?\n[patient] uh , i'm an x-ray tech .\n"
        "[patient] went to [US_STATE] last weekend to hike near the mount baker area .\n"
        "NOTE: The patient has been seen at Point May Orthopedics. Date of birth [DOB] 1982.")


def _ext(**fields):
    empty = {"value": None, "evidence": None, "rationale": None}
    base = {f: dict(empty) for f in ("age", "birth_year", "sex", "state_or_region", "city", "occupation",
                                       "employer_or_facility", "ethnicity", "nationality", "family_structure")}
    return base | {"absolute_dates": [], "other_facts": []} | fields


def _attr(value, evidence):
    return {"value": value, "evidence": evidence, "rationale": "r"}


def test_exact_and_near_quotes_accepted():
    kept, dropped, n = verify(_ext(
        occupation=_attr("x-ray tech", "i'm an x-ray tech."),  # punctuation spacing differs
        employer_or_facility=_attr("Point May Orthopedics", "has been seen at Point May Orthopedic"),  # near quote
    ), TEXT)
    assert n == 2 and not dropped and {a["field"] for a in kept} == {"occupation", "employer_or_facility"}


def test_invented_quote_and_placeholder_value_rejected():
    kept, dropped, n = verify(_ext(
        city=_attr("Seattle", "she lives in downtown Seattle near the market"),
        state_or_region=_attr("[US_STATE]", "went to [US_STATE] last weekend"),
        occupation=_attr("teacher", None),
    ), TEXT)
    assert not kept and n == 3
    assert {d["field"]: d["reason"] for d in dropped} == {
        "city": "evidence_not_found", "state_or_region": "placeholder_value", "occupation": "missing_evidence"}


def test_inferred_value_rejected_but_visible_residue_kept():
    kept, dropped, _ = verify(_ext(
        state_or_region=_attr("Washington", "hike near the mount baker area"),
        birth_year=_attr("[DOB] 1982", "Date of birth [DOB] 1982"),
    ), TEXT)
    assert [d["reason"] for d in dropped] == ["value_not_in_evidence"]
    assert kept[0]["value"] == "1982"


def test_clean_value_keeps_bracket_detail():
    assert clean_value("[LOCATION: Middle East]") == "Middle East"
    assert clean_value("[DATE] 2021") == "2021"


@pytest.mark.parametrize("ph", [
    "[PATIENT_NAME]", "[CLINICIAN_NAME]", "[CLINICIAN_NAME_1]", "[PERSON]", "[PERSON_2]", "[RELATIVE_NAME]",
    "[DOB]", "[DATE]", "[MONTH]", "[WEEKDAY]", "[HOLIDAY]", "[ID]", "[US_STATE]", "[LOCATION]",
    "[LOCATION: Middle East]", "[ORGANIZATION]", "[HEALTHCARE_FACILITY]", "[EMPLOYER]", "[OCCUPATION]",
    "[ETHNICITY]", "[NATIONALITY]", "[ADDRESS]"])
def test_placeholder_regex_matches_all_types(ph):
    assert PLACEHOLDER_RE.fullmatch(ph)
    assert PLACEHOLDER_RE.findall(f"x {ph} 1982 y") == [ph]


@pytest.mark.parametrize("text", ["[40-50]", "[<5.7]", "[>=240]", "[doctor]", "[patient]", "90+"])
def test_placeholder_regex_ignores_generalized_values_and_speakers(text):
    assert not PLACEHOLDER_RE.search(text)
