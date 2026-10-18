"""Step 4: approximate US population frequencies that turn an attribute into a probability p.

Sources (rounded on purpose: the score needs orders of magnitude, not exact shares):
- Age: US Census Bureau, ACS 2020 5-year estimates, population by 5-year age group (table S0101).
- State: US Census 2020 resident population by state, plus DC.
- Ethnicity: Census 2020 race and Hispanic origin; national-origin groups from ACS 2020 ancestry
  (B04006) and Hispanic/Asian detailed groups; Jewish share from Pew Research 2020.
- Occupation: BLS Occupational Employment and Wage Statistics (OEWS), May 2023 national estimates;
  p = occupation employment / total employment (151.9M). Farmers and the military are not in OEWS
  (BLS Employment Projections and DoD counts instead).
City, employer, nationality, region, dates, family structure and rarity buckets are heuristic
constants or buckets, flagged as such by p_source.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

# --- age -----------------------------------------------------------------------------------
CENSUS_YEAR = 2020
AGE_MAX = 104
# (start of 5-year group, % of US population); 85+ split by the ACS 85-89/90-94/95+ detail.
_AGE_GROUPS = [(0, 6.0), (5, 6.1), (10, 6.5), (15, 6.5), (20, 6.6), (25, 7.0), (30, 6.8), (35, 6.5),
               (40, 6.1), (45, 6.2), (50, 6.3), (55, 6.6), (60, 6.3), (65, 5.3), (70, 4.3), (75, 2.9),
               (80, 1.9), (85, 1.15), (90, 0.57), (95, 0.16), (100, 0.02)]
_AGE_TOTAL = sum(s for _, s in _AGE_GROUPS)
AGE_SHARE = {start + k: share / _AGE_TOTAL / 5 for start, share in _AGE_GROUPS for k in range(5)}


def age_range_share(lo: int, hi: int) -> float:
    return sum(AGE_SHARE.get(a, 0.0) for a in range(max(lo, 0), min(hi, AGE_MAX) + 1)) or AGE_SHARE[AGE_MAX]


def parse_age(value: str) -> tuple[int, int] | None:
    """'50' -> (50, 50); '90+' -> (90, 104); '40-50' -> (40, 50); 'late 60s' -> (66, 69)."""
    s = value.lower()
    if m := re.search(r"(\d{1,3})\s*\+|(?:over|older than|above)\s+(\d{1,3})", s):
        return int(m.group(1) or m.group(2)), AGE_MAX
    if m := re.search(r"(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\b", s):
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        return lo, hi
    if m := re.search(r"(early|mid|late)?[\s-]*(\d)0'?s\b", s):
        d = int(m.group(2)) * 10
        return {"early": (d, d + 3), "mid": (d + 3, d + 6), "late": (d + 6, d + 9)}.get(m.group(1), (d, d + 9))
    if re.search(r"\b(months?|weeks?|days?)\b", s) and "year" not in s:
        return 0, 0
    if m := re.search(r"\d{1,3}", s):
        return int(m.group()), int(m.group())
    return None


# --- geography -------------------------------------------------------------------------------
STATE_POP = {  # Census 2020, millions
    "california": 39.54, "texas": 29.15, "florida": 21.54, "new york": 20.20, "pennsylvania": 13.00,
    "illinois": 12.81, "ohio": 11.80, "georgia": 10.71, "north carolina": 10.44, "michigan": 10.08,
    "new jersey": 9.29, "virginia": 8.63, "washington": 7.71, "arizona": 7.15, "massachusetts": 7.03,
    "tennessee": 6.91, "indiana": 6.79, "maryland": 6.18, "missouri": 6.15, "wisconsin": 5.89,
    "colorado": 5.77, "minnesota": 5.71, "south carolina": 5.12, "alabama": 5.02, "louisiana": 4.66,
    "kentucky": 4.51, "oregon": 4.24, "oklahoma": 3.96, "connecticut": 3.61, "utah": 3.27, "iowa": 3.19,
    "nevada": 3.10, "arkansas": 3.01, "mississippi": 2.96, "kansas": 2.94, "new mexico": 2.12,
    "nebraska": 1.96, "idaho": 1.84, "west virginia": 1.79, "hawaii": 1.46, "new hampshire": 1.38,
    "maine": 1.36, "rhode island": 1.10, "montana": 1.08, "delaware": 0.99, "south dakota": 0.89,
    "north dakota": 0.78, "alaska": 0.73, "district of columbia": 0.69, "vermont": 0.64, "wyoming": 0.58,
}
STATE_ABBR = dict(zip(
    "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd "
    "oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy".split(),
    ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
     "district of columbia", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
     "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
     "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
     "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
     "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
     "west virginia", "wisconsin", "wyoming"], strict=True))
_STATES_LONGEST_FIRST = sorted(STATE_POP, key=len, reverse=True)
_US_NAMES = {"united states", "usa", "us", "u s", "u s a", "america", "the us", "the united states", "the states"}
REGION_P = 0.2
CITY_P = 1e-3


def canonical_state(value: str) -> str | None:
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", value.lower())) + " "
    if re.search(r" (washington )?d ?c |district of columbia", t):
        return "district of columbia"
    for name in _STATES_LONGEST_FIRST:
        if f" {name} " in t:
            return name
    return STATE_ABBR.get(value.strip().lower().replace(".", ""))


def place_p(value: str) -> tuple[float, str, str]:
    """state -> census share; the US as a whole -> 1; any other region or country -> 0.2."""
    if state := canonical_state(value):
        return STATE_POP[state] / sum(STATE_POP.values()), "census", state
    norm = re.sub(r"[^a-z ]", " ", value.lower()).split()
    if " ".join(norm) in _US_NAMES:
        return 1.0, "constant", "united states"
    return REGION_P, "constant", " ".join(norm)


# --- occupation --------------------------------------------------------------------------------
TOTAL_EMPLOYMENT_K = 151_900
OCCUPATIONS_K = {  # BLS OEWS May 2023, thousands employed (rounded)
    "retail salesperson": 3700, "home health aide": 3700, "fast food worker": 3600, "general manager": 3400,
    "cashier": 3300, "teacher": 3300, "registered nurse": 3200, "laborer": 3000,
    "customer service representative": 2900, "stocker": 2800, "office clerk": 2500, "waiter": 2200,
    "truck driver": 2100, "janitor": 2100, "assembler": 1900, "administrative assistant": 1800,
    "software developer": 1700, "engineer": 1700, "bookkeeper": 1600, "maintenance worker": 1500,
    "cook": 1400, "accountant": 1400, "elementary school teacher": 1400, "sales representative": 1400,
    "nursing assistant": 1350, "teaching assistant": 1300, "professor": 1300, "security guard": 1100,
    "retail supervisor": 1100, "receptionist": 1000, "construction worker": 1000, "high school teacher": 1000,
    "delivery driver": 1000, "farmer": 900, "landscaper": 900, "management analyst": 900,
    "project manager": 900, "housekeeper": 880, "electrician": 760, "medical assistant": 740,
    "physician": 750, "financial manager": 750, "carpenter": 730, "lawyer": 730, "social worker": 720,
    "mechanic": 670, "police officer": 650, "licensed practical nurse": 630, "middle school teacher": 600,
    "bartender": 600, "childcare worker": 500, "plumber": 480, "insurance agent": 480, "bus driver": 470,
    "pharmacy technician": 450, "preschool teacher": 450, "welder": 420, "farmworker": 400,
    "machinist": 360, "bank teller": 360, "hairdresser": 350, "personal trainer": 340, "pharmacist": 330,
    "mail carrier": 330, "civil engineer": 330, "firefighter": 320, "construction manager": 310,
    "mechanical engineer": 290, "nurse practitioner": 280, "paramedic": 270, "physical therapist": 240,
    "radiologic technologist": 220, "dental hygienist": 210, "graphic designer": 210,
    "electrical engineer": 190, "real estate agent": 170, "chef": 150, "physician assistant": 150,
    "occupational therapist": 140, "librarian": 130, "dentist": 120, "flight attendant": 120,
    "airline pilot": 110, "veterinarian": 85, "photographer": 55, "writer": 50, "actor": 50, "musician": 40,
}
STATUS_P = {"military": 1.3e6 / 151.9e6, "student": 0.06, "retired": 0.17, "unemployed": 0.04,
            "homemaker": 0.04, "disabled": 0.03, "business owner": 0.1}  # constants (DoD, CPS, rough)
OWNER_OF_SPECIFIC_BUSINESS_P = 1e-3  # "owns an auto repair shop": a few 100k such businesses
UNKNOWN_OCCUPATION_P = 1e-2
EMPLOYED_P = 0.6  # BLS employment-population ratio (~60%)
ALIASES = {
    "x-ray tech": "radiologic technologist", "x-ray technician": "radiologic technologist",
    "xray tech": "radiologic technologist", "x ray tech": "radiologic technologist",
    "radiology tech": "radiologic technologist", "radiology technician": "radiologic technologist",
    "rad tech": "radiologic technologist", "radiographer": "radiologic technologist",
    "nurse": "registered nurse", "rn": "registered nurse", "lpn": "licensed practical nurse",
    "cna": "nursing assistant", "doctor": "physician", "md": "physician", "waitress": "waiter",
    "server": "waiter", "trucker": "truck driver", "cdl driver": "truck driver", "programmer": "software developer",
    "software engineer": "software developer", "developer": "software developer", "cop": "police officer",
    "police": "police officer", "secretary": "administrative assistant", "auto mechanic": "mechanic",
    "car mechanic": "mechanic", "warehouse worker": "laborer", "factory worker": "assembler",
    "caregiver": "home health aide", "personal care aide": "home health aide", "custodian": "janitor",
    "cleaner": "housekeeper", "maid": "housekeeper", "hairstylist": "hairdresser", "barber": "hairdresser",
    "cosmetologist": "hairdresser", "attorney": "lawyer", "cpa": "accountant", "emt": "paramedic",
    "fireman": "firefighter", "postal worker": "mail carrier", "mailman": "mail carrier",
    "uber driver": "delivery driver", "rancher": "farmer", "soldier": "military", "army": "military",
    "marine": "military", "navy": "military", "air force": "military", "college student": "student",
    "grad student": "student", "stay at home mom": "homemaker", "housewife": "homemaker", "retiree": "retired",
    "line cook": "cook", "pilot": "airline pilot", "construction": "construction worker",
    "self employed": "business owner", "small business owner": "business owner",
}
SOC_MAJOR_GROUPS_K = [  # keyword -> BLS OEWS May 2023 major-group employment (thousands); first match wins
    (r"nurse|physician|surgeon|therapist|technologist|pharmac|dentist|hygienist|paramedic|clinician|radiolog|sonograph|dietitian|optometrist|chiropractor", "healthcare practitioners", 9400),
    (r"aide|caregiver|orderly|phlebotom|massage|medical assistant|health ?care worker|infusion", "healthcare support", 7300),
    (r"teacher|professor|instructor|tutor|educator|librarian|principal|lecturer|coach", "education", 8800),
    (r"engineer|architect|drafter|surveyor", "architecture and engineering", 2600),
    (r"software|programmer|developer|computer|\bit\b|network|web|data|statistic|actuar", "computer and mathematical", 5200),
    (r"lawyer|attorney|paralegal|judge|legal", "legal", 1200),
    (r"scientist|chemist|biologist|physicist|research|laborator|geologist|economist|psycholog", "science", 1400),
    (r"social worker|counselor|pastor|priest|minister|clergy|chaplain", "community and social service", 2400),
    (r"manager|director|executive|\bceo\b|owner|administrator|supervisor|president", "management", 10500),
    (r"accountant|auditor|financ|banker|broker|consultant|underwriter|insurance|human resources|recruiter|analyst|planner", "business and financial", 10100),
    (r"artist|designer|musician|singer|writer|editor|journalist|photographer|actor|dancer|athlete|reporter|producer", "arts and media", 2100),
    (r"police|officer|guard|firefighter|sheriff|correction|security|detective", "protective service", 3500),
    (r"cook|chef|waiter|waitress|server|bartender|barista|dishwasher|food|restaurant|baker", "food preparation and serving", 13400),
    (r"janitor|custodian|cleaner|housekeep|maid|landscap|groundskeep|pest", "building and grounds cleaning", 4500),
    (r"hair|barber|stylist|nail|trainer|childcare|nanny|funeral", "personal care and service", 3300),
    (r"sales|cashier|retail|real estate|realtor", "sales", 13300),
    (r"clerk|secretary|receptionist|assistant|office|admin|bookkeep|dispatcher|teller|postal|mail|customer service|desk|computer", "office and administrative support", 18500),
    (r"farm|ranch|fisher|logger|forestry", "farming, fishing and forestry", 450),
    (r"construction|carpenter|electrician|plumber|roofer|mason|painter|laborer|miner|drywall|ironworker|contractor", "construction and extraction", 6300),
    (r"mechanic|repair|technician|installer|hvac|maintenance|lineman", "installation, maintenance and repair", 6000),
    (r"factory|machinist|assembl|welder|production|operator|manufactur|plant|printer|butcher", "production", 8800),
    (r"driver|pilot|trucker|delivery|warehouse|forklift|mover|courier|flight attendant|conductor", "transportation and material moving", 13900),
]
_OCC_KEYS = list(OCCUPATIONS_K) + list(ALIASES) + list(STATUS_P)
_QUALIFIERS = r"\b(retired|former|ex|part[- ]time|full[- ]time|senior|junior|lead|head|works? as|i m|im|is|was|an?|the)\b"


def occupation_p(value: str) -> tuple[float, str, str]:
    if re.fullmatch(r"\s*(?:is |currently )?(?:working|employed|works|has a job|full[- ]time|part[- ]time)\s*", value.lower()):
        return EMPLOYED_P, "bls", "employed (no occupation given)"
    if re.search(r"from home|remote", value.lower()):
        return 0.15, "table", "works from home (ACS 2022)"
    raw = re.sub(r"[^a-z\- ]", " ", value.lower())
    s = re.sub(r"\s+", " ", re.sub(_QUALIFIERS, " ", raw)).strip() or raw.strip()
    if re.search(r"\b(owns?|owner|owning)\b", s) and not re.fullmatch(r"(small )?business owner|owner", s):
        return OWNER_OF_SPECIFIC_BUSINESS_P, "constant", f"owner: {s}"
    key = ALIASES.get(s, s)
    if key not in OCCUPATIONS_K and key not in STATUS_P:
        best = process.extractOne(s, _OCC_KEYS, scorer=fuzz.token_sort_ratio, score_cutoff=88)
        key = ALIASES.get(best[0], best[0]) if best else key
    if key in STATUS_P:
        return STATUS_P[key], "constant", key
    if key in OCCUPATIONS_K:
        return OCCUPATIONS_K[key] / TOTAL_EMPLOYMENT_K, "bls", key
    for pattern, group, employed_k in SOC_MAJOR_GROUPS_K:
        if re.search(pattern, s):
            return employed_k / TOTAL_EMPLOYMENT_K, "bls", f"SOC group: {group}"
    return UNKNOWN_OCCUPATION_P, "constant", s


# --- other categories --------------------------------------------------------------------------
ETHNICITY = {  # Census 2020 / ACS 2020 shares of the US population
    "white": .578, "caucasian": .578, "hispanic": .187, "latino": .187, "latina": .187, "latinx": .187,
    "black": .121, "african american": .124, "asian": .059, "asian american": .059, "native american": .009,
    "american indian": .009, "alaska native": .009, "pacific islander": .002, "native hawaiian": .002,
    "multiracial": .041, "mixed race": .041, "biracial": .041, "mexican": .112, "puerto rican": .017,
    "cuban": .007, "dominican": .006, "salvadoran": .007, "guatemalan": .005, "chinese": .016,
    "asian indian": .014, "indian": .014, "filipino": .013, "vietnamese": .0067, "korean": .0057,
    "japanese": .0046, "german": .13, "irish": .097, "english": .076, "italian": .051, "polish": .027,
    "french": .023, "scottish": .016, "jewish": .024, "arab": .011, "middle eastern": .011,
    "haitian": .003, "jamaican": .003, "nigerian": .001, "ethiopian": .0008, "somali": .0005,
}
_ETHNICITY_LONGEST_FIRST = sorted(ETHNICITY, key=len, reverse=True)
UNKNOWN_ETHNICITY_P = 0.05


def ethnicity_p(value: str) -> tuple[float, str, str]:
    s = " " + re.sub(r"[^a-z ]", " ", value.lower()) + " "
    for key in _ETHNICITY_LONGEST_FIRST:
        if f" {key} " in s:
            return ETHNICITY[key], "census", key
    return UNKNOWN_ETHNICITY_P, "constant", value.lower()


def nationality_p(value: str) -> tuple[float, str, str]:
    if re.search(r"\b(americans?|us citizen|u\.?s\.?a?|united states)\b", value.lower()):
        return 1.0, "constant", "american"
    return 0.1, "constant", value.lower()


_MONTH = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)\.?"


def date_p(value: str) -> tuple[float, str]:
    """Precision of a date: day -> 1/365, month -> 1/12, year only -> 1 (no information alone)."""
    s = value.lower()
    if (re.search(rf"\b{_MONTH}\s+\d{{1,2}}(st|nd|rd|th)?\b", s) or re.search(rf"\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{_MONTH}", s)
            or re.search(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b", s) or re.search(r"\b\d{4}-\d{2}-\d{2}\b", s)):
        return 1 / 365, "day"
    if re.search(rf"\b{_MONTH}\b", s) or re.search(r"\b\d{1,2}/\d{4}\b", s):
        return 1 / 12, "month"
    return 1.0, "year" if re.search(r"\b(19|20)\d{2}\b", s) else "unparsed"


_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())} | {"dozen": 12}


def _first_number(s: str) -> int | None:
    m = re.search(r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\b", s)
    return None if not m else int(m.group()) if m.group().isdigit() else _NUMBER_WORDS[m.group()]


def family_p(value: str) -> tuple[float, str]:
    """Rough buckets for family size and household composition (default 1e-2)."""
    s, n = value.lower(), _first_number(value.lower())
    if "triplet" in s:
        return 1e-3, "triplet"
    if "twin" in s:
        return 0.03, "twin"
    if n is not None and re.search(r"grand(child|kid|son|daughter)", s):
        return (1e-2 if n >= 10 else 0.1), f"{n} grandchildren"
    if n is not None and re.search(r"sibling|brother|sister|of us", s):
        n -= 1 if "of us" in s else 0  # "ten of us" = nine siblings
        return (1e-3 if n >= 8 else 1e-2 if n >= 5 else 0.1 if n >= 3 else 0.5), f"{n} siblings"
    if n is not None and re.search(r"child|kid|son|daughter", s):
        return (1e-3 if n >= 7 else 1e-2 if n >= 5 else 0.05 if n == 4 else 0.1 if n == 3 else 0.4), f"{n} children"
    for pattern, p, label in ((r"expecting|pregnant|newborn", 0.02, "expecting a child"), (r"adopt", 0.02, "adopted"),
                              (r"\bhas (?:a |little |young |grown |two |)(?:children|kids?|child|sons?|daughters?)\b",
                               0.6, "has children"),
                              (r"\bhas an? (?:sister|brother|sibling)", 0.8, "has a sibling"),
                              (r"lives with (?:his |her |their )?(?:father|mother|dad|mom|parents)", 0.3, "lives with parents"),
                              (r"girlfriend|boyfriend|lady friend|significant other", 0.3, "has a partner"),
                              (r"widow", 0.06, "widowed"), (r"divorc", 0.1, "divorced"),
                              (r"lives? alone", 0.12, "lives alone"), (r"\bsingle\b", 0.3, "single"),
                              (r"roommate|housemate", 0.08, "lives with roommates"),
                              (r"married|wife|husband|spouse|partner", 0.5, "married")):
        if re.search(pattern, s):
            return p, label
    return 1e-2, "other"


# Common lifestyle facts with a published prevalence (US, approximate shares of the population; rounded):
# priced from this table instead of the LLM's rarity bucket, which overrates everyday facts.
# Sources: handedness literature (~90% right-handed); FHWA licensed drivers (~70% of the population);
# CDC NHIS 2021 (current smokers 11.5%, former 22%); Gallup 2023 (drinks alcohol 62%); CDC sleep (~65% sleep
# >= 7 h); CDC exercise guidelines met (~25%); SFIA 2023 participation (hiking ~18%, running ~15%, biking ~15%,
# swimming ~10%, yoga ~10%, tennis ~7%, basketball ~8%, soccer ~4%, skiing ~3%); NGF 2023 golf (~8%); APPA
# (dog ~40%, cat ~25% of people in owning households); VA (veterans ~5.4%); ACS 2022 (works from home ~15%,
# public transit commute ~3% of workers -> ~2% of people, renter-occupied multi-unit housing ~30%, lives alone ~12%);
# CDC (muscle-strengthening ~30% -> 20% regular; leisure walking ~50%); NHIS 2017 (meditation 14%); RBFF (fishing ~16%);
# BLS ATUS (long working hours ~15% of workers); SFIA (pilates ~3%); AHS (basement ~35% of homes, single-family
# ~65%); Census CPS (movers ~10%/yr); NGA (gardening ~35% of people); Gallup (sports fans ~40%, watches ~50%).
LIFESTYLE = [  # (regex on the fact, p, label); first match wins, so specific entries come first
    (r"left[- ]handed", 0.10, "left-handed"), (r"right[- ]handed", 0.90, "right-handed"),
    (r"\bquit smoking|former smoker|ex-smoker|used to smoke", 0.22, "former smoker"),
    (r"\bsmok|cigarette|\bvap(?:e|es|ing)\b|tobacco", 0.12, "current smoker"),
    (r"\b(?:drinks?|drinker|beers?|wine|alcohol|liquor)\b", 0.60, "drinks alcohol"),
    (r"\bsleeps?\b|hours of sleep|eight hours", 0.65, "sleeps 7+ hours"),
    (r"\bveteran\b|\bserved in the (?:army|navy|marines|air force)", 0.054, "veteran"),
    (r"work(?:s|ing)? from home|remote work", 0.15, "works from home"),
    (r"\bbus\b|subway|public transport|\btrain\b to work", 0.02, "public transit"),
    (r"\bdrives?\b|\bdriving\b|owns? a car|has a car", 0.70, "drives / owns a car"),
    (r"\bgolf", 0.08, "golf"), (r"\bhik(?:e|es|ing)\b", 0.18, "hiking"), (r"\btennis\b", 0.07, "tennis"),
    (r"\bski(?:s|ing|er)?\b", 0.03, "skiing"), (r"\bbasketball\b", 0.08, "basketball"), (r"\bsoccer\b", 0.04, "soccer"),
    (r"\byoga\b", 0.10, "yoga"), (r"\bswim", 0.10, "swimming"), (r"\b(?:run|runs|running|jog|jogs|jogging)\b", 0.15, "running"),
    (r"\b(?:bike|bikes|biking|cycling|cycles)\b", 0.15, "cycling"),
    (r"\b(?:gym|exercis\w*|work(?:s|ing)? out)\b", 0.25, "exercises regularly"),
    (r"\blifts?\b.*\bweights?\b|weight ?lifting|strength training", 0.20, "strength training"),
    (r"\bpilates\b", 0.03, "pilates"), (r"\bfish(?:es|ing)?\b", 0.15, "fishing"),
    (r"\bmeditat", 0.14, "meditation"), (r"\bwalk(?:s|ing)?\b", 0.50, "walks for exercise"),
    (r"long hours|late nights|overtime", 0.15, "long working hours"),
    (r"\bdogs?\b", 0.40, "dog"), (r"\bcats?\b", 0.25, "cat"),
    (r"\bapartment\b|\bcondo\b", 0.30, "lives in an apartment"),
    (r"\bbasement\b", 0.35, "home with a basement"), (r"lives in a house\b|single[- ]family", 0.65, "lives in a house"),
    (r"\bwatch(?:es|ed|ing)? (?:football|sports|games?|tv|television|the masters)", 0.50, "watches sports/TV"),
    (r"\b(?:sports|football|baseball|basketball) fan\b", 0.40, "sports fan"), (r"sedentary", 0.25, "sedentary"),
    (r"\b(?:moved|relocated)\b", 0.10, "moved in the last year"), (r"\bgarden", 0.35, "gardening"),
    (r"yard ?work|works? in the yard|\bmows?\b", 0.40, "yard work"),
]


def lifestyle_p(value: str) -> tuple[float, str] | None:
    s = value.lower()
    for pattern, p, label in LIFESTYLE:
        if re.search(pattern, s):
            return p, label
    return None


RARITY_P = {"common": 1e-1, "uncommon": 1e-2, "rare": 1e-3, "very_rare": 1e-4}
NAMED_EMPLOYER_P = 1e-4
GENERIC_EMPLOYER_P = 0.1
_GENERIC_ORG_WORDS = set(
    "a an the my his her their our local big small large private public state city county federal government "
    "hospital clinic school company plant factory warehouse store restaurant office firm bank army navy church "
    "university college practice grocery retail construction family business farm nursing home center medical "
    "health dental law shop auto repair car dealership airline post fire police department military base va "
    "orthopedic orthopedics pediatric physical therapy rehab urgent care er emergency room".split())


def employer_p(value: str) -> tuple[float, str]:
    words = re.findall(r"[a-z]+", value.lower())
    if words and all(w in _GENERIC_ORG_WORDS for w in words):
        return GENERIC_EMPLOYER_P, "generic"
    return NAMED_EMPLOYER_P, "named"
