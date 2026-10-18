"""Guards for rules the extractor model breaks, and union scoring of the original."""
import pytest

from attack.run import analyze
from attack.score import score_record
from attack.verify import NAME_RE, is_medical, verify

TEXT = ("[patient] i went to washington to hike around mount baker . i race pigeons . "
        "i got a kidney transplant . Daniel Moore is a 62-year-old male.")


def _ext(**fields):
    base = {f: {"value": None, "evidence": None, "rationale": None} for f in (
        "age", "birth_year", "sex", "state_or_region", "city", "occupation", "employer_or_facility",
        "ethnicity", "nationality", "family_structure")}
    return base | {"absolute_dates": [], "other_facts": []} | fields


def _fact(value, evidence, rarity="rare"):
    return {"value": value, "evidence": evidence, "rationale": "r", "rarity": rarity}


@pytest.mark.parametrize("value,expected", [
    ("kidney transplant recipient", True), ("hypothyroidism", True), ("back pain", True),
    ("COVID-19 vaccinated", True), ("melanoma", True), ("painter", False), ("high school diploma", False),
    ("owns an auto repair shop", False), ("veteran with 1% disability", False), ("races pigeons", False),
])
def test_medical_filter(value, expected):
    assert is_medical(value) is expected


def test_medical_filter_uses_icd_descriptions():
    assert is_medical("polycystic kidneys", ("Polycystic kidney, unspecified",))


@pytest.mark.parametrize("value,expected", [
    ("Daniel Moore", True), ("Mrs. Jones", False), ("Mount Baker", False), ("Point May Orthopedics", False),
    ("Army veteran", False), ("Helen", False), ("New York", False),
])
def test_name_filter(value, expected):
    assert bool(NAME_RE.fullmatch(value)) is expected


def test_other_facts_guards_drop_without_counting_as_hallucination():
    ext = _ext(other_facts=[_fact("kidney transplant", "i got a kidney transplant"),
                            _fact("Daniel Moore", "Daniel Moore is a 62-year-old male"),
                            _fact("races pigeons", "i race pigeons", "very_rare")])
    kept, dropped, n = verify(ext, TEXT)
    assert [a["value"] for a in kept] == ["races pigeons"] and n == 3
    assert {d["reason"] for d in dropped} == {"medical_fact", "name_not_scored"}


def test_state_in_other_facts_is_priced_as_state():
    kept, _, _ = verify(_ext(other_facts=[_fact("hikes in Washington", "i went to washington to hike")]), TEXT)
    attr = score_record(kept)["attributes"][0]
    assert attr["category"] == "state_or_region" and attr["p_source"] == "census" and 0.02 < attr["p"] < 0.03


def test_original_scores_union_of_both_extractions():
    rec = {"encounter_id": "E1", "text": TEXT, "masking_density": 0.0, "n_placeholders": 0, "n_icd_codes": 0,
           "icd_descriptions": (), "icd_external": ()}
    orig_ext = _ext(age={"value": "62", "evidence": "62-year-old male", "rationale": "r"})
    anon_ext = _ext(age={"value": "62", "evidence": "62-year-old male", "rationale": "r"},
                    other_facts=[_fact("races pigeons", "i race pigeons", "very_rare"),
                                 _fact("invented", "this quote is not in the text")])
    alone, union = analyze(rec, orig_ext), analyze(rec, orig_ext, anon_ext, "from_anonymized")
    assert union["bits"] == pytest.approx(alone["bits"] + 13.2877, abs=1e-3)
    assert union["n_extracted"] == alone["n_extracted"] == 1 and not union["dropped"]  # own stats only
    assert [a["value"] for a in union["attributes"] if a.get("from_anonymized")] == ["races pigeons"]


# --- regressions from the first real five-record run (qwen3.5:35b on D2N001-D2N005) -------------
@pytest.mark.parametrize("value", ["systolic ejection murmur", "has atrial fibrillation", "due for mammogram"])
def test_medical_filter_real_run_misses(value):
    assert is_medical(value)


def test_name_hint_and_relative_dates_are_dropped():
    text = "[patient] my full name is peter lang . see you last weekend . born in march 1980 ."
    ext = _ext(absolute_dates=[{"value": "last weekend", "evidence": "see you last weekend", "rationale": "r"},
                               {"value": "march 1980", "evidence": "born in march 1980", "rationale": "r"}],
               other_facts=[_fact("full name Peter Lang", "my full name is peter lang", "very_rare")])
    kept, dropped, _ = verify(ext, text)
    assert [a["value"] for a in kept] == ["march 1980"]
    assert sorted(d["reason"] for d in dropped) == ["name_not_scored", "not_absolute_date"]


def test_restated_other_fact_is_counted_once():
    facts = [dict(_fact("hikes regularly", "i've been doing a lot of hiking", "common"), field="other_facts"),
             dict(_fact("hiking", "i've been doing a lot of hiking", "common"), field="other_facts"),
             dict(_fact("races pigeons", "i race pigeons", "very_rare"), field="other_facts")]
    attrs = score_record(facts)["attributes"]
    assert {a["value"] for a in attrs if a["counted"]} == {"races pigeons", "hikes regularly"}
    assert next(a for a in attrs if a["value"] == "hiking")["duplicate_of"] == "hikes regularly"


def test_recovery_scoring_real_run_false_positives():
    from attack.recover import score_guesses
    guesses = {"guesses": [
        {"placeholder": "[PATIENT_NAME]", "type": "name", "guess": "John Doe", "reasoning": "", "confidence": "low"},
        {"placeholder": "[DOB]", "type": "year", "guess": "1973", "reasoning": "", "confidence": "medium"},
        {"placeholder": "[WEEKDAY]", "type": "year", "guess": "Saturday", "reasoning": "", "confidence": "low"},
        {"placeholder": "[US_STATE]", "type": "state_or_region", "guess": "WA", "reasoning": "", "confidence": "high"},
    ]}
    truth = {"[PATIENT_NAME]": ["john", "Mr John Baxter"], "[WEEKDAY]": ["saturday"], "[US_STATE]": ["washington"]}
    present = {"[PATIENT_NAME]", "[WEEKDAY]", "[US_STATE]"}
    s = {g["placeholder"]: g for g in score_guesses(guesses, truth, [], present)}
    assert s["[PATIENT_NAME]"]["scorable"] and not s["[PATIENT_NAME]"]["correct"]  # filler name
    assert s["[DOB]"]["truth_source"] == "placeholder_not_in_text" and not s["[DOB]"]["scorable"]
    assert s["[WEEKDAY]"]["truth_source"] == "out_of_scope" and not s["[WEEKDAY]"]["scorable"]
    assert s["[US_STATE]"]["correct"]


def test_anonymized_record_gets_original_facts_only_when_quoted_verbatim():
    anon_text = "[patient] i went to [US_STATE] to hike . we keep moving boxes up the basement stairs ."
    rec = {"encounter_id": "E1", "text": anon_text, "masking_density": 1.0, "n_placeholders": 1, "n_icd_codes": 0,
           "icd_descriptions": (), "icd_external": ()}
    orig_ext = _ext(other_facts=[_fact("house with a basement", "moving boxes up the basement stairs", "uncommon"),
                                 _fact("hikes in Washington", "i went to washington to hike", "uncommon")])
    row = analyze(rec, _ext(), orig_ext, "from_original", fuzzy_also=False)
    assert [a["value"] for a in row["attributes"]] == ["house with a basement"]  # the masked state can't leak in
    assert row["attributes"][0]["from_original"] and row["n_extracted"] == 0


# --- tuning after the human review of the five records -------------------------------------------
def test_facts_restating_external_cause_codes_are_floor():
    from attack.data import to_record
    rec = to_record({"encounter_id": "E", "dialogue": "[patient] i hurt my back golfing .", "note": "",
                     "icd10_codes": [{"code": "S39.012A", "description": "Strain of muscle of lower back"},
                                     {"code": "Y93.53", "description": "Activity, golf"}]})
    assert rec["icd_external"] == ("Activity, golf",)
    ext = _ext(other_facts=[_fact("plays golf", "i hurt my back golfing", "common")])
    kept, dropped, _ = verify(ext, rec["text"], rec["icd_descriptions"], icd_external=rec["icd_external"])
    assert not kept and dropped[0]["reason"] == "icd_floor"


def test_region_placeholders_match_by_name():
    from attack.recover import is_match
    assert is_match("state_or_region", "North Cascades", "cascades") is True
    assert is_match("state_or_region", "Bellingham", "cascades") is False
    assert is_match("state_or_region", "Washington", "cascades") is False  # a state is not the region


def test_recovery_summary_separates_names():
    from attack.recover import summarize_recovery
    g = {"placeholder": "[X]", "reasoning": "", "confidence": "high", "truth": [], "truth_source": "alignment",
         "scorable": True}
    rows = [{"encounter_id": "E", "recovery": [dict(g, type="name", guess="Mary", correct=False),
                                               dict(g, type="state_or_region", guess="WA", correct=True)]}]
    s = summarize_recovery(rows)
    assert s["overall"]["recovery_rate"] == 0.5 and s["excluding_names"]["recovery_rate"] == 1.0


def test_elided_quotes_are_checked_fragment_by_fragment():
    from attack.verify import normalize, quote_in_text
    text = normalize("[doctor] you have somewhat of a sweet tooth . [patient] i love jelly beans , yeah .")
    assert quote_in_text("you have somewhat of a sweet tooth ... i love jelly beans", text)
    assert not quote_in_text("you have somewhat of a sweet tooth ... i hate licorice", text)


@pytest.mark.parametrize("fact,expected", [("rear-ended at a four-way stop", True), ("drives a truck", True),
                                           ("races pigeons", False), ("personal trainer", False)])
def test_vehicle_accident_code_covers_synonyms(fact, expected):
    from attack.verify import restates_icd
    assert restates_icd(fact, ("Person injured in unspecified motor-vehicle accident, traffic",)) is expected


# --- fixes from the 67-record evaluation run ----------------------------------------------------
def test_no_control_characters_in_sources():
    from pathlib import Path
    for f in Path(__file__).resolve().parents[1].glob("*.py"):
        assert "\b" not in f.read_text(), f  # a "\b" written without a raw string


@pytest.mark.parametrize("fact,p", [("right-handed", 0.90), ("drives a car", 0.70), ("smokes 1.5 packs per day", 0.12),
                                    ("quit smoking for almost a year", 0.22), ("plays golf on weekends", 0.08),
                                    ("veteran with 1% disability", 0.054), ("races pigeons", None)])
def test_lifestyle_table(fact, p):
    from attack.tables import lifestyle_p
    hit = lifestyle_p(fact)
    assert (hit[0] if hit else None) == p


def test_family_facts_in_other_facts_are_priced_and_deduplicated_as_family():
    attrs = score_record([
        {"field": "family_structure", "value": "ten siblings", "evidence": "there is actually ten of us", "rationale": "r"},
        {"field": "other_facts", "value": "has ten siblings", "evidence": "ten of us", "rationale": "r", "rarity": "rare"},
        {"field": "other_facts", "value": "lives with wife", "evidence": "my wife", "rationale": "r", "rarity": "common"},
    ])["attributes"]
    assert [a["category"] for a in attrs] == ["family_structure"] * 3
    assert sum(a["counted"] for a in attrs) == 1  # one family value, the most specific


def test_other_fact_restating_occupation_is_not_counted_twice():
    attrs = score_record([
        {"field": "occupation", "value": "lecturer", "evidence": "i'm a lecturer", "rationale": "r"},
        {"field": "other_facts", "value": "works as a lecturer", "evidence": "a lecturer at the college", "rationale": "r",
         "rarity": "uncommon"},
        {"field": "other_facts", "value": "rides a road bike 30 miles", "evidence": "i ride 30 miles", "rationale": "r",
         "rarity": "uncommon"}])["attributes"]
    by = {a["value"]: a for a in attrs}
    assert by["works as a lecturer"]["duplicate_of"] == "lecturer" and not by["works as a lecturer"]["counted"]
    assert by["rides a road bike 30 miles"]["counted"]


@pytest.mark.parametrize("value,expected", [("had aortic valve replacement", True), ("had cholecystectomy", True),
                                            ("lost 110 pounds intentionally", True), ("uses THC gummies and cream", True),
                                            ("owns an auto repair shop", False), ("rides a road bike", False)])
def test_medical_filter_procedures_weight_supplements(value, expected):
    assert is_medical(value) is expected


def test_inferred_place_in_other_facts_is_not_priced_as_a_state():
    text = "[patient] went to [US_STATE] to hike near mount baker ."
    ext = _ext(other_facts=[_fact("hiked near Mount Baker in Washington", "to hike near mount baker", "uncommon")])
    kept, _, _ = verify(ext, text)
    assert kept[0]["inferred_place"] == "washington"
    assert score_record(kept)["attributes"][0]["category"] == "other_facts"


@pytest.mark.parametrize("value", ["has pursed lip breathing", "takes clindamycin prior to dental work",
                                   "entered menopause at age 28", "takes metformin"])
def test_medical_filter_eval_misses(value):
    assert is_medical(value)


def test_family_history_of_disease_is_medical():
    ext = _ext(family_structure={"value": "mother, maternal aunt and grandmother", "evidence": "Mother: melanoma",
                                 "rationale": "r"})
    kept, dropped, _ = verify(ext, "FAMILY HISTORY\nMother: melanoma.")
    assert not kept and dropped[0]["reason"] == "medical_fact"


def test_paraphrased_other_facts_are_deduplicated():
    attrs = score_record([
        {"field": "other_facts", "value": "takes nature photographs", "evidence": "i take pictures of nature", "rarity": "uncommon"},
        {"field": "other_facts", "value": "enjoys nature photography", "evidence": "Hobbies: photography", "rarity": "uncommon"}])["attributes"]
    assert sum(a["counted"] for a in attrs) == 1


@pytest.mark.parametrize("text,value", [("so gloria is a 46 -year-old female today", "46"),
                                        ("[PERSON] is a [40-49] -year-old male", "40-49"),
                                        ("is a 40 49 year old female", "40-49"), ("an 80 year old white male", "80"),
                                        ("my 45-year-old husband drives", None), ("smoked for 20 years", None)])
def test_age_fallback(text, value):
    from attack.verify import fallback_age
    hit = fallback_age(text)
    assert (hit["value"] if hit else None) == value


def test_age_fallback_only_when_llm_found_no_age():
    rec = {"encounter_id": "E", "text": "[doctor] gloria is a 46 -year-old female .", "masking_density": 0,
           "n_placeholders": 0, "n_icd_codes": 0, "icd_descriptions": (), "icd_external": ()}
    row = analyze(rec, _ext())
    assert [(a["value"], a.get("fallback")) for a in row["attributes"]] == [("46", True)]
    row = analyze(rec, _ext(age={"value": "46", "evidence": "46 -year-old female", "rationale": "r"}))
    assert [a.get("fallback") for a in row["attributes"]] == [None]


def test_fact_about_a_masked_referent_is_dropped():
    ext = _ext(other_facts=[_fact("traveled to a US state to explore", "went up to [US_STATE] um to go um explore",
                                  "uncommon")])
    kept, dropped, _ = verify(ext, "[patient] i went up to [US_STATE] um to go um explore the mountains .")
    assert not kept and dropped[0]["reason"] == "masked_referent"


# --- general fixes after the round-1 human review (eval, age_bands) ------------------------------
@pytest.mark.parametrize("value,p", [("has children", 0.6), ("has a sister", 0.8), ("has a kid", 0.6),
                                     ("lives with father", 0.3), ("expecting a daughter", 0.02), ("ten siblings", 1e-3)])
def test_generic_family_facts(value, p):
    from attack.tables import family_p
    assert family_p(value)[0] == p


@pytest.mark.parametrize("value,p", [("working", 0.6), ("employed", 0.6), ("works from home", 0.15)])
def test_non_informative_occupations(value, p):
    from attack.tables import occupation_p
    assert occupation_p(value)[0] == p


@pytest.mark.parametrize("fact,label", [("social drinker only", "drinks alcohol"), ("rear-ended in a car accident", None),
                                        ("lives in a house with basement stairs", "home with a basement")])
def test_lifestyle_table_round2(fact, label):
    from attack.tables import lifestyle_p
    hit = lifestyle_p(fact)
    assert (hit[1] if hit else None) == label


def test_same_lifestyle_theme_counted_once():
    attrs = score_record([
        {"field": "other_facts", "value": "smokes 1.5 packs per day", "evidence": "a pack and a half a day", "rarity": "common"},
        {"field": "other_facts", "value": "started smoking in high school", "evidence": "since high school", "rarity": "common"},
    ])["attributes"]
    assert sum(a["counted"] for a in attrs) == 1


@pytest.mark.parametrize("value", ["has epilepsy", "has gallstones", "had lumbar fusion", "nauseous in the car",
                                   "sees a counselor once a week", "uses a wheelchair", "uses T-Gel shampoo", "has PCOS"])
def test_medical_vocabulary_round2(value):
    assert is_medical(value)


def test_accident_detail_restates_vehicle_code():
    from attack.verify import restates_icd
    assert restates_icd("stopped at a four-way stop when hit", ("Person injured in unspecified motor-vehicle accident",))


# --- recovery scoring fixes ---------------------------------------------------------------------
def test_recovery_prompt_has_no_values_from_the_evaluation_set():
    from attack.recover import SYSTEM_PROMPT
    for leaked in ("1982", "Washington", "Seattle"):  # v3 examples quoted evaluation records
        assert leaked not in SYSTEM_PROMPT


def test_repeated_placeholder_is_scored_against_its_most_frequent_value():
    from attack.recover import score_guesses
    truth = {"[DATE]": ["march 2019", "april 2020", "may 2020", "june 2020"]}
    rec = {"guesses": [{"placeholder": "[DATE]", "type": "year", "guess": "2019", "reasoning": "", "confidence": "high"}]}
    assert score_guesses(rec, truth, [], {"[DATE]"})[0]["correct"] is False  # 2019 is not the dominant value
    rec["guesses"][0]["guess"] = "2020"
    assert score_guesses(rec, truth, [], {"[DATE]"})[0]["correct"] is True


def test_one_scored_guess_per_placeholder():
    from attack.recover import score_guesses
    g = {"placeholder": "[ORGANIZATION]", "type": "employer_or_facility", "reasoning": ""}
    rec = {"guesses": [dict(g, guess="Acme", confidence="low"), dict(g, guess="Initech", confidence="high")]}
    out = score_guesses(rec, {"[ORGANIZATION]": ["initech", "initech"]}, [], {"[ORGANIZATION]"})
    assert [(x["guess"], x["truth_source"]) for x in out] == [("Initech", "alignment"), ("Acme", "duplicate_guess")]
    assert sum(x["scorable"] for x in out) == 1
