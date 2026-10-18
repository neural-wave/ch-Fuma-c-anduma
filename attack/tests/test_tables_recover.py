import pytest

from attack import tables
from attack.recover import align_placeholders, is_match, score_guesses, summarize_recovery


def test_occupation_lookup():
    p, source, canon = tables.occupation_p("x-ray technician")
    assert canon == "radiologic technologist" and source == "bls" and p == pytest.approx(220 / 151_900)
    assert tables.occupation_p("owns an auto repair shop")[0] == tables.OWNER_OF_SPECIFIC_BUSINESS_P
    assert tables.occupation_p("retired teacher")[2] == "teacher"
    assert tables.occupation_p("hvac installer")[2].startswith("SOC group")
    assert tables.occupation_p("zeppelin whisperer")[0] == tables.UNKNOWN_OCCUPATION_P


def test_geography_age_dates_family_employer():
    assert tables.canonical_state("WA") == tables.canonical_state("northern Washington") == "washington"
    assert tables.canonical_state("West Virginia") == "west virginia"
    assert tables.place_p("California")[0] == pytest.approx(39.54 / sum(tables.STATE_POP.values()))
    assert tables.place_p("Middle East")[0] == tables.REGION_P
    assert 0.012 < tables.age_range_share(50, 50) < 0.015
    assert tables.parse_age("90+") == (90, tables.AGE_MAX)
    assert tables.parse_age("in her late 60s") == (66, 69)
    assert tables.date_p("march 3rd, 2021")[1] == "day"
    assert tables.date_p("march 2019")[1] == "month"
    assert tables.date_p("2021")[0] == 1.0
    assert tables.family_p("ten siblings")[0] == 1e-3
    assert tables.family_p("there are ten of us")[0] == 1e-3
    assert tables.employer_p("Point May Orthopedics")[1] == "named"
    assert tables.employer_p("the local hospital")[1] == "generic"


def test_alignment_recovers_masked_spans():
    anon = ("went to [US_STATE] last weekend to hike in northern [LOCATION], like around mount baker . "
            "hi [PATIENT_NAME] .")
    orig = "went to washington last weekend to hike in northern washington , like around mount baker . hi mrs. jones ."
    truth = align_placeholders(anon, orig)
    assert truth["[US_STATE]"] == ["washington"] and truth["[LOCATION]"] == ["washington"]
    assert truth["[PATIENT_NAME]"] == ["mrs jones"]


@pytest.mark.parametrize("kind,guess,truth,expected", [
    ("age", "40", "39-year-old", True), ("age", "42", "39", False),
    ("state_or_region", "WA", "washington", True), ("state_or_region", "Oregon", "washington", False),
    ("year", "2021", "march 3rd, 2021", True), ("year", "2021", "next week", None),
    ("name", "Jones", "mrs. jones", True), ("name", "Dr.", "dr. smith", False),
    ("employer_or_facility", "the hospital", "st. mary's hospital", False),
    ("employer_or_facility", "Point May Orthopedic", "point may orthopedics", True),
])
def test_matching_rules(kind, guess, truth, expected):
    assert is_match(kind, guess, truth) is expected


def test_score_and_summarize_guesses():
    recovery = {"guesses": [
        {"placeholder": "[US_STATE]", "type": "state_or_region", "guess": "Washington", "reasoning": "mount baker",
         "confidence": "high"},
        {"placeholder": "[DATE]", "type": "year", "guess": "2021", "reasoning": "dob + age", "confidence": "high"},
        {"placeholder": "[CITY]", "type": "city", "guess": "Bellingham", "reasoning": "near baker",
         "confidence": "low"},
    ]}
    truth = {"[US_STATE]": ["washington"], "[DATE]": ["next week"]}
    orig_attrs = [{"field": "city", "value": "seattle"}]
    scored = score_guesses(recovery, truth, orig_attrs)
    assert [(g["scorable"], g["correct"]) for g in scored] == [(True, True), (False, False), (True, False)]
    assert scored[2]["truth_source"] == "original_extraction"
    s = summarize_recovery([{"encounter_id": "D2N001", "recovery": scored}])
    assert s["overall"]["recovery_rate"] == pytest.approx(0.5)
    assert s["best_recoveries"][0]["guess"] == "Washington"
    assert s["unverified_high_confidence"][0]["placeholder"] == "[DATE]"
