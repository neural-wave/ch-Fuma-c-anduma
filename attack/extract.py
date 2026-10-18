"""Step 2: one structured LLM call per record listing the patient's quasi-identifiers."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from .data import NOTE_SEPARATOR
from .llm import LLMClient, inline_refs

PROMPT_VERSION = "extract-v4"


class Attr(BaseModel):
    value: str | None
    evidence: str | None  # verbatim quote from the text
    rationale: str | None  # one line: why this narrows down who the patient is


class OtherFact(Attr):
    rarity: Literal["common", "uncommon", "rare", "very_rare"]


class Extraction(BaseModel):
    age: Attr
    birth_year: Attr
    sex: Attr
    state_or_region: Attr
    city: Attr
    occupation: Attr
    employer_or_facility: Attr
    ethnicity: Attr
    nationality: Attr
    absolute_dates: list[Attr]  # any date with a year or a month
    family_structure: Attr  # e.g. "10 siblings", "5 children"
    other_facts: list[OtherFact]  # hobbies, owns a business, veteran, rare events...


SCALAR_FIELDS = ("age", "birth_year", "sex", "state_or_region", "city", "occupation",
                 "employer_or_facility", "ethnicity", "nationality", "family_structure")
SCHEMA = inline_refs(Extraction.model_json_schema())

SYSTEM_PROMPT = """You are a privacy auditor running a re-identification attack on a de-identified medical encounter (a doctor-patient dialogue followed by the clinical note). List every quasi-identifier the text still reveals about THE PATIENT, so we can measure how identifiable they are.

Rules:
1. Only facts about the patient. Ignore clinicians and other people; family_structure is the patient's own family or household. Speaker tags ([doctor], [patient]) can be wrong or swapped: decide who the patient is from the content.
2. UPPERCASE text in square brackets ([PATIENT_NAME], [US_STATE], [DATE], [ORGANIZATION], ...) is a masked value. Never output a placeholder or its label as a value; a fact given only as a placeholder is unknown (null). Visible residue is a stated value: "[DOB] 1982" gives birth_year "1982"; "[LOCATION: Middle East]" gives state_or_region "Middle East".
3. Generalized values: extract what is actually stated, e.g. age "90+" or "40-50", or a year.
4. Not stated means null (value, evidence and rationale all null). Do not guess or infer: a landmark does not state a state or a city.
5. Every non-null value needs evidence: a verbatim quote of 5 to 12 words copied character for character from the text (keep its spacing and punctuation) and containing the value; one continuous passage, never fragments joined with "...". The value is the fact in short form, worded as in the text.
6. rationale: one short line on why the fact narrows down who the patient is.
7. state_or_region / city: where the patient lives, works or comes from. Travel destinations, landmarks and neighborhoods go to other_facts.
8. absolute_dates: dates with a month or a year ("march 2019", "june 3rd", "in 2015"); not relative times ("two weeks ago") or weekdays. The birth year goes to birth_year only.
9. other_facts: non-medical facts that would help find the patient. Check each area: work (duties, workplace, schedule, even without a job title), housing, transport, hobbies and sports, travel, pets, possessions, military service, and notable or recent events (an accident, a move, a wedding, an arrest). Describe the fact itself, e.g. "was rear-ended in a car accident 4-5 days ago", not "drives a car". rarity = share of the US population with the fact: common = about 1 in 10, uncommon = 1 in 100, rare = 1 in 1,000, very_rare = 1 in 10,000 or rarer. Calibrate by area. Work: a broad line of work (office or finance work, healthcare, construction) is common, a specific job (crane operator on night shifts) uncommon. Housing and transport: a walk-up apartment, taking the bus, living on a farm are common. Hobbies: golf, hiking, fishing are common, marathons uncommon, pigeon racing very_rare. Travel: a trip to a state or large region is common, to a named landmark or small town uncommon. Events: a recent accident, move, wedding or arrest is uncommon, and rare only with specific details (exact place and circumstances). Army veteran is common. Skip facts shared by more than about 1 in 10 people (has a basement, likes sweets, orders takeout, lives with family, owns a car). Never include names, diagnoses, symptoms, medications, vaccinations, procedures, test results or vital signs, and never repeat a fact already captured in another field.
10. Return only JSON matching the schema."""


def _a(value=None, evidence=None, rationale=None, **extra) -> dict:
    return {"value": value, "evidence": evidence, "rationale": rationale, **extra}


def _answer(**fields) -> str:
    out = {f: _a() for f in SCALAR_FIELDS} | {"absolute_dates": [], "other_facts": []} | fields
    return json.dumps(Extraction.model_validate(out).model_dump(), separators=(",", ":"))


FEW_SHOTS = [
    (
        "[doctor] hi [PATIENT_NAME] , what brings you in today ?\n"
        "[patient] my lower back . i drive a forklift at the [EMPLOYER] warehouse and it's been bad since march 2019 .\n"
        "[doctor] what do you do to relax ?\n"
        "[patient] i race pigeons , believe it or not . and we have five kids , so not much relaxing ."
        + NOTE_SEPARATOR
        + "HISTORY OF PRESENT ILLNESS\n[PATIENT_NAME] is a 58-year-old male from Ohio, date of birth [DOB] 1963, "
          "with chronic low back pain. He is an Army veteran.",
        _answer(
            age=_a("58", "58-year-old male from Ohio", "Exact age: a one-year cohort."),
            birth_year=_a("1963", "date of birth [DOB] 1963", "Birth year left visible next to the masked date."),
            sex=_a("male", "58-year-old male", "Halves the candidates."),
            state_or_region=_a("Ohio", "58-year-old male from Ohio", "State of residence."),
            occupation=_a("forklift operator", "i drive a forklift", "Occupation."),
            absolute_dates=[_a("march 2019", "it's been bad since march 2019", "Month and year of onset.")],
            family_structure=_a("five children", "we have five kids", "Large family."),
            other_facts=[
                _a("races pigeons", "i race pigeons", "Very unusual hobby.", rarity="very_rare"),
                _a("Army veteran", "He is an Army veteran", "Veteran status.", rarity="common"),
            ],
        ),
    ),
    (
        "[doctor] hello [PATIENT_NAME] , how is the cough ?\n"
        "[patient] still there . it started [DATE] , right after my trip to [LOCATION] .\n"
        "[doctor] any fever or chills ?\n"
        "[patient] no , just tired . my a1c came back [<5.7] . carrying laundry down to the basement wiped me out ."
        + NOTE_SEPARATOR
        + "CHIEF COMPLAINT\nCough.\n\nHISTORY OF PRESENT ILLNESS\n[PATIENT_NAME] presents with 3 weeks of dry cough. "
          "She denies fever.",
        _answer(sex=_a("female", "She denies fever", "Halves the candidates.")),
    ),
    (
        "[doctor] are you still working ?\n"
        "[patient] part-time . i teach piano at [ORGANIZATION] .\n"
        "[patient] and where are you living now ?\n"
        "[doctor] a retirement village near lake tahoe . my wife passed in [MONTH] , so i live alone ."
        + NOTE_SEPARATOR
        + "SOCIAL HISTORY\nThe patient is a 90+ year-old Hispanic man, widowed, born in [NATIONALITY], "
          "moved to the US in 1975.",
        _answer(
            age=_a("90+", "a 90+ year-old Hispanic man", "Generalized age, still a small cohort."),
            sex=_a("male", "Hispanic man", "Halves the candidates."),
            occupation=_a("piano teacher", "i teach piano", "Uncommon occupation."),
            ethnicity=_a("Hispanic", "90+ year-old Hispanic man", "Ethnicity."),
            absolute_dates=[_a("1975", "moved to the US in 1975", "Year of immigration.")],
            family_structure=_a("widowed, lives alone", "my wife passed in [MONTH] , so i live alone",
                                "Household composition."),
            other_facts=[_a("lives in a retirement village near Lake Tahoe", "a retirement village near lake tahoe",
                            "Small geographic area.", rarity="rare")],
        ),
    ),
]


def build_messages(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, answer in FEW_SHOTS:
        messages += [{"role": "user", "content": example}, {"role": "assistant", "content": answer}]
    return messages + [{"role": "user", "content": text}]


def extract(client: LLMClient, text: str) -> dict:
    return client.chat_json(build_messages(text), SCHEMA, "extraction", PROMPT_VERSION,
                            lambda d: Extraction.model_validate(d).model_dump())
