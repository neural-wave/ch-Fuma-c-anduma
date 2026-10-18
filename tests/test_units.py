"""Fast unit tests for lexical utilities, rules and transformations (no spaCy model needed)."""
import pytest

from medanon.lexicon import load_lexicon, number_to_words, words_to_number
from medanon.rules import RuleDetector
from medanon.transform import age_band, perturb_lab


@pytest.fixture(scope="module")
def rules():
    return RuleDetector(load_lexicon())


def labels(rules, text, field="dialogue"):
    return {(s.label, text[s.start:s.end]) for s in rules.detect(text, field)}


@pytest.mark.parametrize("text,value", [("62", 62), ("sixty two", 62), ("sixty-two", 62), ("eight point three", 8.3),
                                        ("one hundred and two", 102), ("seven", 7)])
def test_words_to_number(text, value):
    assert words_to_number(text) == value


def test_number_to_words_roundtrip():
    for v, d in [(8.1, 1), (62, 0), (145, 0), (6.5, 1)]:
        assert words_to_number(number_to_words(v, d)) == v


def test_name_ratio_separates_names_from_words():
    lex = load_lexicon()
    assert lex.name_ratio("andrew") > 0.5 and lex.name_ratio("perez") > 0.5
    assert lex.name_ratio("will") < 0.1 and lex.name_ratio("hope") < 0.1 and lex.name_ratio("may") < 0.1


def test_drug_lexicon_is_not_polluted_by_places_or_names():
    lex = load_lexicon()
    for w in ("arizona", "johnson", "timothy", "kroger", "target"):
        assert not lex.is_drug(w), w
    for w in ("metformin", "lisinopril", "eliquis", "lasix"):
        assert lex.medical_strength(w) == 2, w


@pytest.mark.parametrize("text,expected", [
    ("[doctor] he is a 62-year-old male .", ("AGE", "62")),
    ("[doctor] she is a sixty two year old female .", ("AGE", "sixty two")),
    ("[patient] i'll be 71 next month .", ("AGE", "71")),
    ("[patient] i turned 50 in october .", ("AGE", "50")),
    ("[doctor] date of birth 3/14/1958 , mrn 4-4-7-1-9-2 .", ("DOB", "3/14/1958")),
    ("[doctor] date of birth , february 15th , 1961 .", ("DOB", "february 15th , 1961")),
    ("[doctor] date of birth 3/14/1958 , mrn 4-4-7-1-9-2 .", ("ID", "4-4-7-1-9-2")),
    ("[doctor] kim king , dob 1/2/1960 , mrnq27631 .", ("ID", "mrnq27631")),
    ("[doctor] we'll see you on march 3rd .", ("DATE", "march 3rd")),
    ("[patient] back in 2015 i had surgery .", ("YEAR", "2015")),
    ("[patient] it started around christmas .", ("HOLIDAY", "christmas")),
])
def test_rule_patterns(rules, text, expected):
    assert expected in labels(rules, text)


@pytest.mark.parametrize("text,name", [
    ("[doctor] good morning , mr. kowalski .", "mr. kowalski"),
    ("[patient] my wife , jenny , drove me .", "jenny"),
    ("[patient] my dog max keeps me active .", "max"),
    ("[doctor] hi andrew , how are you ?", "andrew"),
    ("[doctor] hey dragon , i'm seeing will kowalski today .", "will kowalski"),
    ("[doctor] hope whitfield , date of birth 2/2/1950 .", "hope whitfield"),
])
def test_context_names(rules, text, name):
    assert ("PERSON", name) in labels(rules, text)


@pytest.mark.parametrize("text", ["[patient] i will take the pills .", "[doctor] we hope it helps .",
                                  "[doctor] hey dragon , show me the x-ray .", "[patient] may i ask something ?"])
def test_no_false_names(rules, text):
    assert not any(lab == "PERSON" for lab, _ in labels(rules, text))


def test_places_institutions_employers(rules):
    t = "[patient] i grew up in ohio and my granddaughter goes to ohio state university . i work as a plumber at home depot ."
    got = labels(rules, t)
    assert ("ORGANIZATION", "ohio state university") in got
    assert ("ORGANIZATION", "home depot") in got
    assert ("OCCUPATION", "plumber") in got
    assert ("LOCATION", "ohio") in got


def test_employer_cue_is_not_too_loose(rules):
    got = labels(rules, "[doctor] keep working with your strategies , and work out at the gym .")
    assert not any(lab == "ORGANIZATION" for lab, _ in got)


def test_lab_value_detection_digits_and_words(rules):
    got = rules.detect("[doctor] your a1c was eight point three and your ldl is 131 .", "dialogue")
    vals = {(s.meta["lab"], s.meta["value"]) for s in got if s.label == "LAB_VALUE"}
    assert ("a1c", 8.3) in vals and ("ldl", 131.0) in vals
    got = rules.detect("[doctor] let's recheck your a1c in three months .", "dialogue")
    assert not any(s.label == "LAB_VALUE" for s in got)


@pytest.mark.parametrize("age,width,band", [(62, 10, "60-69"), (95, 10, "90+"), (3, 10, "1-4"), (62, 20, "60-79")])
def test_age_band(age, width, band):
    assert age_band(age, width) == band


def test_perturbation_never_crosses_clinical_thresholds():
    from medanon.rules import LABS
    import bisect

    for lab, val, dec in [("a1c", 8.3, 1), ("a1c", 6.5, 1), ("a1c", 7.0, 1), ("glucose", 126, 0), ("ldl", 99, 0)]:
        for seed in range(30):
            nv = perturb_lab(lab, val, dec, f"s{seed}", 0.03)
            th = LABS[lab][1]
            if nv is not None:
                assert nv != val and bisect.bisect_right(th, nv) == bisect.bisect_right(th, val)
