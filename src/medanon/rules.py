"""High-precision rules and context patterns.

They are designed for the two text styles in the data: cased clinical notes and *lowercased*
ASR-style dialogues (where capitalisation-based NER fails). Every span carries the rule that
fired and a human-readable reason, which feeds the explanation log.
"""
from __future__ import annotations

import re

from .lexicon import (
    RACE_ETHNICITY, CLINICIAN_TITLES, DECIMAL_WORDS_RE, FULL_MONTHS, GREETINGS, HOLIDAYS, MEDICAL_HEADS, MONTHS, NOT_NAMES,
    NUM_WORDS_RE, ORDINAL_WORDS_RE, PET_WORDS, POSSESSIVES, RELATIONS, TITLES, WEEKDAYS, Lexicon,
    is_cased, tokenize, words_to_number,
)
from . import quasi
from .spans import Span

I = re.IGNORECASE
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
_FULL_MONTH_ALT = "|".join(sorted(FULL_MONTHS, key=len, reverse=True))
_TENS_ALT = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_UNIT_ALT = "one|two|three|four|five|six|seven|eight|nine"
_TEEN_ALT = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
SPOKEN_YEAR = (rf"(?:(?:nineteen|twenty)\s+(?:(?:{_TENS_ALT})(?:[\s-]+(?:{_UNIT_ALT}))?|{_TEEN_ALT}|oh\s+(?:{_UNIT_ALT}))"
               rf"|two\s+thousand(?:\s+and)?(?:\s+(?:(?:{_TENS_ALT})(?:[\s-]+(?:{_UNIT_ALT}))?|{_TEEN_ALT}|{_UNIT_ALT}))?)")
_DAY = rf"(?:\d{{1,2}}(?:st|nd|rd|th)?|{ORDINAL_WORDS_RE})"
_YEAR = rf"(?:\d{{4}}|'\d{{2}}|{SPOKEN_YEAR})"
DATE_FORMS = [
    r"\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*(?:\d{4}|\d{2})(?!\d)",
    r"(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}(?!\d)",  # month/year: 03/2019
    r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}",
    rf"(?:{_MONTH_ALT})\.?\s+(?:the\s+)?{_DAY}(?:\s*,?\s*{_YEAR})?",
    rf"(?:the\s+)?{_DAY}\s+(?:of\s+)?(?:{_MONTH_ALT})\.?(?:\s*,?\s*{_YEAR})?",
    rf"(?:{_MONTH_ALT})\.?\s*,?\s*(?:of\s+)?(?:\d{{4}}|{SPOKEN_YEAR})",
]
DATE_RE = re.compile(r"\b(?:" + "|".join(DATE_FORMS) + r")\b", I)
DOB_KEY = r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date|birthday|born\s+on)"
DOB_VALUE = ("|".join(DATE_FORMS)
             # compact MMDDYYYY / YYYYMMDD (only after a DOB cue; elsewhere 8 digits may be an identifier)
             + r"|(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:19|20)\d{2}(?!\d)|(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)"
             + r"|\d{1,2}\s*,\s*\d{1,2}\s*,\s*\d{2,4}|\d{4}(?!\d)"
             + rf"|(?:(?:{NUM_WORDS_RE}|{ORDINAL_WORDS_RE}|{_MONTH_ALT})[\s,]+){{1,6}}(?:{SPOKEN_YEAR}|\d{{4}})")
DOB_RE = re.compile(DOB_KEY + r"(?:\s*(?:is|was|of|on|:|,|-|\.))*\s*(?P<val>" + DOB_VALUE + ")", I)
# the date given in the next dialogue turn, answering a date-of-birth question
DOB_ANSWER_RE = re.compile(DOB_KEY + r"[^\n?]*\?[ \t]*\n\[[a-z_]+\][ \t]*(?:(?:it's|it is|it was|um|uh|yes|yeah|yep|sure|so|"
                           r"okay|ok|oh|well|like|you know|that's|that is|let me see|let's see|my birthday is|my birthday's|"
                           r"i was born on|i was born|born on|born)[\s,.]+)*(?P<val>" + DOB_VALUE + ")", I)
BORN_IN_RE = re.compile(rf"\bborn\s+in\s+(?P<val>(?:19|20)\d{{2}}|'\d{{2}}|{SPOKEN_YEAR})\b", I)
YEAR_RE = re.compile(
    r"(?<![\d$.,/-])(?P<y>(?:19|20)\d{2})(?!\d|[.,/]\d|\s*(?:mg|mcg|ml|cc|units?|calories|kcal|steps|feet|ft|"
    r"meters|miles|dollars|lbs?|pounds|grams|g\b|kg|ng|pg|iu|mmol|cells|platelets|k\b))", I)
SPOKEN_YEAR_RE = re.compile(rf"\b(?:in|since|from|until|till|of|year|around|circa|by)\s+(?P<y>{SPOKEN_YEAR})\b", I)
MONTH_ALONE_RE = re.compile(
    rf"\b(?P<prep>in|since|during|until|till|by|from|of|last|this|next|early|late|mid|around|before|after|"
    rf"through|end\s+of|beginning\s+of|middle\s+of)\s+(?P<m>{_FULL_MONTH_ALT})\b", I)
RELATIVE_DATE_RE = re.compile(
    rf"\b(?:yesterday|the\s+day\s+before\s+yesterday|(?:last|this\s+past|past)\s+(?:night|week(?:end)?|month|year|spring|summer|winter|fall|autumn)|"
    rf"(?:\d+|{NUM_WORDS_RE}|a\s+few|a\s+couple(?:\s+of)?|several|a)\s+(?:days?|weeks?|months?|years?)\s+ago|"
    rf"(?:this|last|next|in\s+the|over\s+the|during\s+the|early|late)\s+(?:spring|summer|winter|autumn|fall)(?!\s+(?:from|risk|precautions))|"
    rf"(?:spring|summer|winter|autumn)(?:time)?)\b", I)
MONTH_ANY_RE = re.compile(r"\b(?:january|february|april|june|july|august|september|october|november|december)\b", I)
WEEKDAY_RE = re.compile(rf"\b(?:{'|'.join(WEEKDAYS)})s?\b", I)
HOLIDAY_RE = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in sorted(HOLIDAYS, key=len, reverse=True)) + r")\b", I)

AGE_PATTERNS = [
    ("age-year-old", re.compile(rf"\b(?P<num>\d{{1,3}}|{NUM_WORDS_RE})(?:\s*-\s*|\s+)(?:year|yr)s?(?:\s*-\s*|\s+)old\b", I)),
    ("age-yo", re.compile(r"\b(?P<num>\d{1,3})\s*(?:yo|y/o|y\.o\.?|yrs?\s+old)(?![a-zß-öø-ÿĀ-ɏḀ-ỿ])", I)),
    ("age-keyword", re.compile(rf"(?<!gestational )(?<!bone )\b(?:age|aged)\s*(?:of\s+|is\s+|:\s*)?(?P<num>\d{{1,3}}|{NUM_WORDS_RE})\b(?!\s*(?:weeks?|days?|months?|percent|%))", I)),
    ("age-copula", re.compile(rf"\b(?:i'm|i am|you're|you are|he's|she's|he is|she is|they're|turned|turning|turn|(?:i|he|she|you)'ll be|"
                              rf"(?:i|he|she|you) will be|will be|(?:i|he|she) was|be)\s+(?P<num>\d{{2,3}}|{NUM_WORDS_RE})"
                              rf"(?=\s*(?:[.,?!;]|$|years?\b|this\b|next\b|now\b|and\b|so\b|but\b|in\b|on\b|at\b|last\b|"
                              rf"back\b|today\b|recently\b|already\b|soon\b|when\b|tomorrow\b))", I | re.M)),
]

# a relative mentioned earlier in the same sentence and the age introduced by "at": family history
FAMILY_ONSET_RE = re.compile(
    r"\b(?:father|mother|dad|mom|brother|sister|son|daughter|grandmother|grandfather|grandma|grandpa|aunt|uncle|"
    r"cousin|sibling|parent|parents|niece|nephew|relative|relatives|kin|family\s+members?|grandparents?|twin|"
    r"siblings|cousins|brothers|sisters|step(?:father|mother|brother|sister)|half-(?:brother|sister))\b"
    r"[^.?!\n]*\bat\s+(?:the\s+)?(?:age\s+(?:of\s+)?)?$", I)
OCC_PHRASE_RE = re.compile(
    r"\b(?:work(?:s|ed|ing)?\s+as|job\s+as|employed\s+as|career\s+as|trained\s+as|retired\s+as|occupation\s*(?:is|:))"
    r"\s+(?:an?\s+)?(?P<occ>[^\W\d_][\w\-']*(?:\s+[^\W\d_][\w\-']*){0,2})", I)
# an age question (also with words after it: "how old are you now ?", "you're how old now ?", "what's your age ?") and
# the number that opens the next turn; questions about a relative's age are excluded (family history keeps them)
_LANDMARK_RE = re.compile(r"\b(?:mount|mt\.?|lake)\s+(?P<n>[a-z][a-z'\-]{2,})\b(?!\s+(?:the|a|an|it|on|in|of|up|down)\b)|"
                          r"\b(?P<n2>(?-i:[A-Z][a-z]+))\s+National\s+(?:Park|Forest)\b", re.I)
AGE_ANSWER_RE = re.compile(rf"\b(?:how\s+old\s+(?:are\s+you|is\s+(?:he|she)|you\s+are)|you'?re\s+how\s+old|what(?:'s|\s+is)\s+your\s+age)"
                           rf"(?![^\n?]*\b(?:son|daughter|mother|mom|father|dad|wife|husband|brother|sister|kids?|children|baby|"
                           rf"grand\w+|parents?)\b)[^\n?]{{0,40}}\??\s*(?:\n?\[[a-z_]+\]\s*)?"
                           rf"(?:(?:i'm|i\s+am|i\s+just\s+turned|just\s+turned|turned|uh|um|so|well|oh|okay|like|you\s+know)\s*,?\s*)*"
                           rf"(?P<num>\d{{1,3}}|{NUM_WORDS_RE})\b(?!\s*(?:percent|%|pounds|minutes|days|weeks|months))", I)
DAY_OF_MONTH_RE = re.compile(rf"\b(?:on|by|until|till|since|before|after|starting)\s+the\s+(?P<d>\d{{1,2}}(?:st|nd|rd|th)|{ORDINAL_WORDS_RE})\b"
                             rf"(?=\s*(?:[.,;:?!)]|$)|\s+(?:of\s+(?!(?:the|my|your)\b)[a-z]+|at|and|or|in|when|if|so|because|then|i|we|you|he|she|they)\b)", I | re.M)
ID_KEYED_RE = re.compile(
    r"\b(?:mrn|medical\s+record(?:\s+number)?|record\s+number|chart\s+number|patient\s+id|account(?:\s+(?:number|no\.?))?|"
    r"member\s+id|policy\s+number|insurance\s+(?:id|number)|ssn|social\s+security(?:\s+number)?)\b"
    r"\s*(?:#|:|is|number|no\.?|,|-)*\s*(?P<id>[a-zß-öø-ÿĀ-ɏḀ-ỿ]{0,4}[-/]?\d[\w-]*|\d(?:[\s-]?\d){3,})", I)
# upper-case letter segments joined by hyphens to a digit block: "QX-PT-7781-02", "R4-MRN-100103"
STRUCTURED_ID_RE = re.compile(r"(?<![\w-])[A-Z]{1,4}[0-9]{0,2}(?:-[A-Z]{1,5}[0-9]{0,2}){0,3}-[0-9]{3,10}(?:-[0-9A-Z]{1,6})*(?![\w-])")
# PIN / passcode: a 4-digit secret that would otherwise look like a (kept) calendar year
PIN_RE = re.compile(r"\b(?:pin(?:\s+(?:number|code))?|passcode|access\s+code)\s*(?:#|:|is|number|no\.?|,|-)*\s*"
                    r"(?P<id>\d(?:[\s-]?\d){3,})", I)
# earlier/later values of the same test in one sentence: protected like the first ("8.3%, down from 9.6%")
_TREND_CONNECTOR = re.compile(r"^[^\d\n]{0,30}\b(?:from|previously|prior|vs\.?|versus|compared\s+(?:to|with)|was|were|"
                              r"down|up|then|now|last\s+time)\b[^\d\n]{0,15}$", I)
_NUMBER_CUE_BEFORE = re.compile(
    r"\b(?:room|rm|suite|ste|apartment|apt|unit|bed|box|p\.?\s?o\.?\s+box|ext|extension|pin|passcode|code|"
    r"id|number|no|account|policy|member|badge|locker|floor)\.?\s*(?:#|:|is|number|no\.?)?\s*#?\s*$", I)
_ADDR_FUNCTION_WORDS = {"with", "in", "at", "on", "and", "by", "for", "to", "of", "from", "since", "after",
                        "before", "during", "until", "was", "is", "were", "had", "have"}
ID_GLUED_RE = re.compile(r"\bmrn[a-zß-öø-ÿĀ-ɏḀ-ỿ]{0,3}\d{3,}[\w-]*\b", I)
CONTACT_RES = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("url", re.compile(r"\b(?:https?://|www\.)\S+", I)),
]
LONG_NUMBER_RE = re.compile(r"(?<![\d.,/])\d{7,}(?![\d.,/])")
# identifiers dictated digit by digit ("four five six seven eight nine", "five five five dash one two ...")
_DIGIT_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple|\d)"
_SPOKEN_DIGITS = rf"{_DIGIT_WORD}(?:(?:\s*[,\-]\s*|\s+dash\s+|\s+){_DIGIT_WORD}){{2,}}"
_ID_CUE = (r"mrn|medical\s+record(?:\s+number)?|record\s+number|chart\s+number|patient\s+id|account(?:\s+number)?|"
           r"member\s+id|policy\s+number|insurance\s+(?:id|number)|ssn|social\s+security(?:\s+number)?|zip(?:\s+code)?|"
           r"id\s+number|(?<!phone\s)number")
_CONTACT_CUE = r"phone(?:\s+number)?|cell(?:\s+phone)?(?:\s+number)?|telephone(?:\s+number)?|fax(?:\s+number)?|call\s+me\s+at|reach\s+me\s+at|extension"
SPOKEN_ID_RE = re.compile(
    rf"\b(?:(?P<contact>{_CONTACT_CUE})|(?P<idcue>{_ID_CUE}))\b(?:\s*(?:is|was|it's|its|:|,|-|uh|um|like))*\s+"
    rf"(?P<val>{_SPOKEN_DIGITS})\b", I)
# seven or more single digits in a row are an identifier even without a cue (phone numbers, MRNs)
UNCUED_DIGITS_RE = re.compile(rf"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)(?:(?:\s*[,\-]\s*|\s+)(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)){{6,}}\b", I)
ZIP_RE = re.compile(r"\b(?:zip(?:\s*code)?|postal\s+code)\s*(?:is|:)?\s*(?P<z>\d{5}(?:-\d{4})?)\b", I)
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9'.-]+\s+){1,3}(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|"
    r"court|ct|way|place|pl|terrace|circle|cir|highway|hwy|parkway|pkwy)\b\.?", I)
DIALOGUE_SUBJECT_RE = re.compile(
    rf"(?:^\[[a-z_]+\]\s+|[.,]\s+|\bseeing\s+|\bpatient\s+)(?P<name>[a-zß-öø-ÿĀ-ɏḀ-ỿ][a-zß-öø-ÿĀ-ɏḀ-ỿ'\-]+(?:\s+[a-zß-öø-ÿĀ-ɏḀ-ỿ][a-zß-öø-ÿĀ-ɏḀ-ỿ'\-]+)?)\s+is\s+an?\s+"
    rf"(?:\d{{1,3}}|{NUM_WORDS_RE})[\s-]+(?:year|month|week)", re.M)
NOTE_SUBJECT_RE = re.compile(
    r"(?:^|(?<=[.\n]\s)|(?<=\n))(?P<name>(?:[A-ZÀ-ÖØ-ÞĀ-ɏḀ-ỿ][a-zß-öø-ÿĀ-ɏḀ-ỿ'\-]+\s){0,2}[A-ZÀ-ÖØ-ÞĀ-ɏḀ-ỿ][a-zß-öø-ÿĀ-ɏḀ-ỿ'\-]+)\s+is\s+an?\s+"
    r"(?:\d{1,3}|[a-z\-]+)[\s-]+(?:year|month|week|day)", re.M)

LOC_CUES = {
    "in", "from", "to", "at", "near", "visiting", "visit", "visited", "moved", "move", "live", "lives",
    "lived", "living", "grew", "born", "trip", "vacation", "flew", "fly", "flying", "drove", "drive",
    "driving", "went", "going", "go", "back", "outside", "around", "toward", "towards", "across",
    "through", "returned", "travel", "traveled", "travelled", "traveling", "cruise", "downtown", "of",
}
INSTITUTION_HEADS = {
    "hospital": "healthcare", "clinic": "healthcare", "medical center": "healthcare",
    "health center": "healthcare", "nursing home": "healthcare", "urgent care": "healthcare", "practice": "healthcare",
    "medical group": "healthcare",
    "university": "school", "college": "school", "high school": "school", "middle school": "school",
    "elementary": "school", "elementary school": "school", "academy": "school", "church": "other",
    "county": "location", "airport": "location",
}
_INST_RE = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, INSTITUTION_HEADS), key=len, reverse=True)) + r")\b", I)
EMPLOY_RE = re.compile(r"\b(?:(?:work|works|worked|working|employed|job)\b(?:\s+(?!out\b|up\b|from\s+home\b)[a-z\-']+){0,4}?"
                       r"\s+(?:at|for)|(?:retired|retire)\s+from)\s+(?:the\s+|a\s+|an\s+)?", I)
GENERIC_WORKPLACE = {
    "home", "night", "nights", "day", "days", "weekend", "weekends", "shift", "shifts", "office", "hospital",
    "school", "store", "factory", "warehouse", "restaurant", "bank", "farm", "plant", "mill", "mine",
    "shop", "church", "airport", "gym", "company", "firm", "city", "county", "state", "government", "site",
    "job", "place", "moment", "time", "desk", "computer", "myself", "himself", "herself", "family", "least",
    "all", "it", "this", "that", "them", "him", "her", "me", "you", "us", "years", "while", "same", "times",
    "once", "random", "lunch", "dinner", "breakfast", "work", "night", "nights", "weekends", "least",
    "local", "big", "small", "new", "old", "construction", "post", "military", "army", "navy", "home",
    "a", "an", "the", "my", "his", "our", "their", "your", "some", "one", "two", "and", "or", "but", "so",
    "because", "now", "still", "like", "um", "uh", "since", "when", "full", "part", "least", "first",
    "evaluation", "treatment", "therapy", "rehab", "appointment", "follow", "followup", "visit", "consultation",
    "testing", "surgery", "procedure", "assessment", "check", "checkup", "management", "strategies", "further",
}
ETHNICITY_RE = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, RACE_ETHNICITY), key=len, reverse=True))
                          + r")\b(?=\s+(?:male|female|man|woman|gentleman|lady|boy|girl|patient|individual|person|descent))", I)
HEADER_NAME_RE = re.compile(
    r"(?im)^[ \t]*(?P<key>patient(?:\s+name)?|name|pt|attending(?:\s+physician)?|physician|provider|"
    r"referring(?:\s+physician)?|pcp|primary\s+care(?:\s+physician)?|surgeon|resident|emergency\s+contact|"
    r"next\s+of\s+kin|guardian|spouse)\s*:[ \t]*(?:dr\.?[ \t]+)?"
    r"(?P<name>[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]+(?:[ \t]*,[ \t]*[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]+|[ \t]+[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]\.?(?=[ \t])|[ \t]+[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]+){0,3})")
INSTITUTION_STOP = {"the", "a", "an", "my", "his", "her", "our", "their", "your", "at", "to", "in", "from", "over",
                    "local", "same", "other", "nearby", "community", "general", "regional", "main", "big", "small",
                    "children's", "veterans", "va"}
# "it is called X", "the store is called X": X names a thing (a business), not a person
NAMED_THING_SUBJECTS = {"it", "place", "company", "business", "store", "shop", "firm", "agency", "practice", "clinic",
                        "center", "centre", "cafe", "restaurant", "bakery", "studio", "organization", "organisation",
                        "charity", "group", "school", "facility", "team", "brand", "plant", "factory", "warehouse",
                        "office", "salon", "gym", "farm", "nonprofit", "co-op", "cooperative", "startup", "workshop",
                        "garage", "pharmacy", "library", "employer", "workplace", "job", "work"}
ORG_NAME_SUFFIXES = {"services", "service", "cooperative", "co-op", "coop", "center", "centre", "inc", "llc", "ltd",
                     "co", "company", "corp", "corporation", "group", "associates", "partners", "studio", "studios",
                     "store", "stores", "shop", "bakery", "market", "pharmacy", "foundation", "institute", "library",
                     "academy", "school", "college", "university", "church", "bank", "hotel", "restaurant", "cafe",
                     "gym", "farm", "farms", "works", "industries", "solutions", "systems", "technologies", "labs",
                     "logistics", "supply", "supplies", "motors", "garage", "salon", "clinic", "hospital", "consulting",
                     "construction", "landscaping", "builders", "media", "brewery", "kitchen", "deli", "diner", "mart",
                     "outfitters", "trading", "transport", "insurance", "realty", "properties", "agency", "collective"}
# a digit run read out after one of these cues (in the same or the previous turn) is a phone number
PHONE_CUE_RE = re.compile(r"\b(?:phone|cell(?:\s*phone)?|mobile|telephone|contact\s+number|call\s*back\s+number|"
                          r"number\s+(?:to\s+(?:call|reach|text)|where\s+(?:we|i)\s+can\s+reach)|reach\s+you|text\s+you)\b", I)
PHRASE_BOUNDARY = {
    "and", "or", "but", "so", "because", "as", "with", "on", "in", "for", "to", "from", "of", "since", "when",
    "where", "who", "which", "that", "i", "he", "she", "we", "they", "it", "all", "every", "during", "until",
    "now", "then", "there", "here", "um", "uh", "like", "is", "was", "about", "doing", "full", "your", "my",
    "his", "her", "our", "their", "this", "these", "those", "some", "any", "yourself", "myself",
}
NAME_SLOT_FOLLOWERS = {
    ",", ".", "?", "!", "who", "is", "was", "has", "had", "and", "'s", "keeps", "says", "said", "thinks", "drove",
    "came", "brought", "helps", "helped", "takes", "took", "lives", "works", "passed", "died", "wants", "noticed",
    "told", "made", "gets", "got", "found", "picked", "drives", "comes", "goes", "went", "recommended",
    "suggested", "usually", "always", "also", "just", "really", "picks", "sent", "called", "checks", "visits",
}
OCC_CUE_RE = re.compile(
    r"(?:\b(?:work(?:s|ed|ing)?|job|career|employed|employment|trained|retired|former|formerly|occupation|"
    r"profession)\b[^.?!\n]{0,40}$|\b(?:i'm|i am|he's|she's|he is|she is|they're|was|been|became|become|"
    r"as|is)\s+(?:a|an)\s+(?:[a-z\-']+\s+){0,3}$)", I)
OCC_STOP = {"a", "an", "the", "as", "is", "was", "i'm", "retired", "former", "my", "his", "her", "and", "uh", "um", ","}

LABS = {
    "a1c": (r"hemoglobin\s+a1c|hb\s?a1c|a1c|glycated\s+hemoglobin|glycosylated\s+hemoglobin", [5.7, 6.5, 7.0, 8.0, 9.0, 10.0], (3.5, 20)),
    "glucose": (r"(?:fasting\s+)?(?:blood\s+)?(?:glucose|sugars?)", [54, 70, 100, 126, 140, 180, 200, 250, 300, 400], (20, 1200)),
    "ldl": (r"ldl(?:\s+cholesterol)?", [70, 100, 130, 160, 190], (10, 400)),
    "hdl": (r"hdl", [40, 50, 60], (5, 150)),
    "cholesterol": (r"(?:total\s+)?cholesterol", [200, 240], (60, 500)),
    "triglycerides": (r"triglycerides?", [150, 200, 500, 1000], (20, 5000)),
    "creatinine": (r"creatinine", [0.6, 1.2, 1.5, 2.0, 3.0, 4.0], (0.2, 20)),
    "egfr": (r"e?gfr", [15, 30, 45, 60, 90], (2, 150)),
    "tsh": (r"tsh", [0.1, 0.4, 4.0, 4.5, 10.0], (0.005, 100)),
    "psa": (r"psa", [2.5, 4.0, 10.0, 20.0], (0.01, 1000)),
    "ef": (r"ejection\s+fraction|lvef|ef", [20, 30, 35, 40, 50, 55, 70], (5, 85)),
    "hemoglobin": (r"hemoglobin(?!\s+a1c)|hgb", [7, 8, 10, 12, 13, 13.5, 16, 17.5], (3, 25)),
    "potassium": (r"potassium", [3.0, 3.5, 5.0, 5.5, 6.0], (1.5, 9)),
    "sodium": (r"sodium", [125, 130, 135, 145, 150], (100, 180)),
    "vitamin_d": (r"vitamin\s+d", [12, 20, 30, 100], (3, 200)),
    "inr": (r"inr", [1.1, 2.0, 3.0, 4.0], (0.5, 15)),
    "bnp": (r"(?:nt-?pro)?bnp", [100, 400, 900], (1, 50000)),
}
_LAB_RE = re.compile(r"\b(?:" + "|".join(f"(?P<{k}>{v[0]})" for k, v in LABS.items()) + r")\b", I)
# the value never starts inside a word, and a spoken value ends at a word boundary ("alone" holds no "one")
_LAB_VALUE_RE = re.compile(rf"(?<![\w/.])(?<!\w-)(?P<v>\d+(?:\.\d+)?|(?:{DECIMAL_WORDS_RE})\b)(?![\d/])(?P<after>\s*(?:%|percent|mg/dl|milligrams?)?\s*\S*)", I)
_TIME_UNIT_AFTER = re.compile(r"^\s*(?:-\s*)?(?:days?|weeks?|months?|years?|hours?|minutes?|times|tablets?|pills?|units?|mg\b(?!\s*/)|mcg|o'clock|am\b|pm\b|a\.m|p\.m|or\s)", I)


def _sp(field: str, s: int, e: int, label: str, rule: str, text: str, score: float, reason: str, **meta) -> Span:
    return Span(field, s, e, label, f"rule:{rule}", score, text[s:e], meta=meta, reasons=[reason])


class RuleDetector:
    def __init__(self, lex: Lexicon, cfg: dict | None = None):
        self.lex = lex
        self.cfg = cfg or {}
        occ = sorted(lex.occupations, key=len, reverse=True)
        self._occ_re = re.compile(r"\b(?P<occ>" + "|".join(re.escape(o) for o in occ) + r")(?:s|es)?\b", I)
        places = sorted(lex.places, key=len, reverse=True)
        self._place_re = re.compile(r"\b(?P<p>" + "|".join(re.escape(p) for p in places) + r")\b", I)

    # ------------------------------------------------------------------ public
    def detect(self, text: str, field: str) -> list[Span]:
        cased = is_cased(text)
        toks = tokenize(text)
        spans: list[Span] = []
        spans += self._ages(text, field)
        spans += self._dates(text, field)
        spans += self._ids(text, field)
        spans += self._names(text, field, toks, cased)
        spans += self._places(text, field, toks, cased)
        spans += self._institutions(text, field, toks, cased)
        spans += self._occupations(text, field)
        spans += self._labs(text, field)
        spans += quasi.detect(text, field)  # height / weight / BMI, family size
        for m in ETHNICITY_RE.finditer(text):
            spans.append(_sp(field, m.start(), m.end(), "ETHNICITY", "ethnicity", text, 0.8, "race / ethnicity term"))
        return spans

    # ------------------------------------------------------------------ ages
    def _ages(self, text: str, field: str) -> list[Span]:
        out = []
        for rule, rx in AGE_PATTERNS:
            for m in rx.finditer(text):
                val = words_to_number(m.group("num"))
                if val is None or not (0 < val <= 120) or (rule == "age-copula" and val < 10):
                    continue
                if FAMILY_ONSET_RE.search(text[max(0, m.start() - 80):m.start("num")]):
                    continue  # "father with colon cancer at age 50": a relative's age of onset, clinically needed
                out.append(_sp(field, m.start("num"), m.end("num"), "AGE", rule, text, 0.9,
                               f"age expression ('{rule}')", age=val))
        # bare answer to an age question: "how old are you ?\n[patient] 62 ."
        for m in AGE_ANSWER_RE.finditer(text):
            val = words_to_number(m.group("num"))
            if val is not None and 1 <= val <= 110:
                out.append(_sp(field, m.start("num"), m.end("num"), "AGE", "age-answer", text, 0.85,
                               "number given in answer to an age question", age=val))
        return out

    # ------------------------------------------------------------------ dates
    def _dates(self, text: str, field: str) -> list[Span]:
        out = []
        for m in DOB_RE.finditer(text):
            out.append(_sp(field, m.start("val"), m.end("val"), "DOB", "dob", text, 0.98, "value following a date-of-birth cue"))
        for m in DOB_ANSWER_RE.finditer(text):
            out.append(_sp(field, m.start("val"), m.end("val"), "DOB", "dob-answer", text, 0.95,
                           "date given in answer to a date-of-birth question"))
        for m in BORN_IN_RE.finditer(text):
            out.append(_sp(field, m.start("val"), m.end("val"), "DOB", "born-in", text, 0.9, "birth year"))
        for m in DATE_RE.finditer(text):
            s = m.group(0)
            if s.lower().startswith("may") and not re.search(r"\d(?:st|nd|rd|th)|\d{4}|,|first|second|third|th\b", s, I):
                continue  # "may 5" is usually the modal verb
            nums = re.findall(r"\d+", s)
            if re.fullmatch(r"\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}", s) and not (
                int(nums[0]) <= 31 and int(nums[1]) <= 31 and (int(nums[0]) <= 12 or int(nums[1]) <= 12)):
                continue
            if len(set(re.findall(r"[/.-]", s))) > 1 and not re.search(r"[A-Za-z]", s):
                continue  # mixed separators ("2-7/10" pain score range) are not a calendar date
            if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2}", s):
                continue  # decimals like 1.2.3 versions / values
            if re.match(r"\.\d|\s*(?:g/dl|mg|mmol|meq|%|k/ul|x10)", text[m.end():m.end() + 8], I):
                continue  # "Hemoglobin 11.2-12.4 g/dL": a range of values, not a date
            if re.search(r"\bborn\s+(?:on\s+)?$", text[max(0, m.start() - 12):m.start()], I):
                out.append(_sp(field, m.start(), m.end(), "DOB", "born-date", text, 0.95, "date after 'born'"))
                continue  # "born 03/14/1979" is a date of birth (its year must follow the DOB policy)
            out.append(_sp(field, m.start(), m.end(), "DATE", "date", text, 0.95, "calendar date"))
        for m in DAY_OF_MONTH_RE.finditer(text):
            out.append(_sp(field, m.start("d"), m.end("d"), "DATE", "day-of-month", text, 0.75, "day-of-month reference"))
        for m in YEAR_RE.finditer(text):
            if _NUMBER_CUE_BEFORE.search(text[max(0, m.start() - 22):m.start()]):
                # "room 2014", "extension 1987", "pin 2018": a number that looks like a year, but calendar
                # years are published by policy while these identify a person or a place
                out.append(_sp(field, m.start("y"), m.end("y"), "ID", "number-cue", text, 0.85, "number after an identifier cue"))
                continue
            out.append(_sp(field, m.start("y"), m.end("y"), "YEAR", "year", text, 0.7, "calendar year"))
        for m in SPOKEN_YEAR_RE.finditer(text):
            out.append(_sp(field, m.start("y"), m.end("y"), "YEAR", "spoken-year", text, 0.7, "spoken calendar year"))
        for m in MONTH_ALONE_RE.finditer(text):
            if m.group("m").lower() == "may" and m.group("prep").lower() in {"this", "by", "from", "of", "around", "after", "before"}:
                continue
            out.append(_sp(field, m.start("m"), m.end("m"), "MONTH", "month", text, 0.6, "month name anchoring an event in time"))
        for m in RELATIVE_DATE_RE.finditer(text):  # kept by default (clinical timeline); policy-switchable
            out.append(_sp(field, m.start(), m.end(), "RELATIVE_DATE", "relative-date", text, 0.5,
                           "relative time or season (no calendar anchor)"))
        for m in MONTH_ANY_RE.finditer(text):  # unambiguous month names need no preposition
            out.append(_sp(field, m.start(), m.end(), "MONTH", "month", text, 0.5, "month name anchoring an event in time"))
        for m in WEEKDAY_RE.finditer(text):
            out.append(_sp(field, m.start(), m.end(), "WEEKDAY", "weekday", text, 0.5, "day-of-week anchor"))
        for m in HOLIDAY_RE.finditer(text):
            out.append(_sp(field, m.start(), m.end(), "HOLIDAY", "holiday", text, 0.6, "holiday = implicit calendar date"))
        return out

    # ------------------------------------------------------------------ identifiers / contact
    def _ids(self, text: str, field: str) -> list[Span]:
        out = []
        for m in ID_KEYED_RE.finditer(text):
            out.append(_sp(field, m.start("id"), m.end("id"), "ID", "id-keyed", text, 0.97, "value following an identifier cue (MRN/account/SSN)"))
        for m in STRUCTURED_ID_RE.finditer(text):
            if len(re.findall(r"\d", m.group(0))) >= 4:  # "QX-PT-7781-02"; "CA-125", "COVID-19" have too few digits
                out.append(_sp(field, m.start(), m.end(), "ID", "structured-id", text, 0.85, "structured record identifier"))
        for m in PIN_RE.finditer(text):
            out.append(_sp(field, m.start("id"), m.end("id"), "ID", "pin", text, 0.95, "value following a PIN / passcode cue"))
        for m in ID_GLUED_RE.finditer(text):
            out.append(_sp(field, m.start(), m.end(), "ID", "id-glued", text, 0.95, "medical record number"))
        for kind, rx in CONTACT_RES:
            for m in rx.finditer(text):
                lab = "ID" if kind == "ssn" else "CONTACT"
                out.append(_sp(field, m.start(), m.end(), lab, kind, text, 0.9, f"{kind} pattern", kind=kind))
        for m in LONG_NUMBER_RE.finditer(text):
            out.append(_sp(field, m.start(), m.end(), "ID", "long-number", text, 0.6, "long digit string (possible identifier)"))
        for m in SPOKEN_ID_RE.finditer(text):
            lab = "CONTACT" if m.group("contact") else "ID"
            out.append(_sp(field, m.start("val"), m.end("val"), lab, "spoken-digits", text, 0.95,
                           f"{'contact number' if lab == 'CONTACT' else 'identifier'} dictated digit by digit after a cue",
                           kind="phone" if lab == "CONTACT" else "id"))
        for m in UNCUED_DIGITS_RE.finditer(text):
            phone = PHONE_CUE_RE.search(text[max(0, m.start() - 250):m.start()])  # cue often in the question turn
            out.append(_sp(field, m.start(), m.end(), "CONTACT" if phone else "ID", "spoken-digits", text, 0.8,
                           "digits dictated after a phone cue (this or the previous turn)" if phone else
                           "seven or more digits dictated in a row (phone / record number)", kind="phone" if phone else "id"))
        for m in ZIP_RE.finditer(text):
            out.append(_sp(field, m.start("z"), m.end("z"), "ADDRESS", "zip", text, 0.9, "postal code"))
        for m in ADDRESS_RE.finditer(text):
            words = re.findall(r"[A-Za-z]+", m.group(0))[:-1]
            if any(w.lower() in _ADDR_FUNCTION_WORDS for w in words):
                continue  # "in 2018 with Dr. Kowalczyk": a year and a clinician, not "2018 ... Drive"
            if re.search(r"\bdr\.?$", m.group(0), I) and re.match(r"\.?\s+[A-Z][a-z]", text[m.end():]):
                continue  # "Dr." followed by a name is a title
            out.append(_sp(field, m.start(), m.end(), "ADDRESS", "street", text, 0.85, "street address"))
        return out

    # ------------------------------------------------------------------ names
    def name_ok(self, tok: str, mode: str, cased: bool) -> bool:
        t = tok.lower().removesuffix("'s").strip("'")
        if len(t) < 2 or not (t[0].isalpha() and all(c.isalpha() or c in "'-" for c in t)) \
                or t in NOT_NAMES or t in RELATIONS or t in TITLES:
            return False
        if cased and not tok[:1].isupper():
            return False
        lex = self.lex
        r, z, med = lex.name_ratio(t), lex.zipf(t), lex.medical_strength(t)
        in_lex = lex.in_name_lexicon(t)
        if mode == "title":
            return (in_lex and z < 5.6) or (z < 3.5 and med < 2)
        if mode == "lenient":
            return (in_lex and r >= 0.1 and med < 2) or (z < 3.0 and med == 0)
        return r >= 0.5 and med < 2  # strict

    def _names(self, text: str, field: str, toks: list, cased: bool) -> list[Span]:
        out: list[Span] = []
        n = len(toks)
        low = [t[0].lower() for t in toks]
        is_word = [bool(re.match(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]", t[0])) for t in toks]

        def word_at(j):
            return j < n and is_word[j]

        def emit(i0, i1, rule, score, reason, role=None, strong=False):
            s, e = toks[i0][1], toks[i1][2]
            name_toks = [low[k].removesuffix("'s") for k in range(i0, i1 + 1)
                         if is_word[k] and low[k].rstrip(".") not in TITLES]
            if name_toks:
                out.append(_sp(field, s, e, "PERSON", rule, text, score, reason, role_hint=role,
                               strong=strong, name_tokens=name_toks))

        def extend_surname(j, mode="strict"):
            """Extend a detected first name by a following surname ("jane doe"): any non-common
            census surname qualifies, otherwise the token must pass the given name test."""
            if not word_at(j + 1) or low[j + 1] in GREETINGS or low[j + 1] in NOT_NAMES:
                return j
            nxt = low[j + 1]
            if (nxt in self.lex.surnames and self.lex.zipf(nxt) < 4.0 and self.lex.medical_strength(nxt) < 2
                    and (not cased or toks[j + 1][0][:1].isupper())) or self.name_ok(toks[j + 1][0], mode, cased):
                return j + 1
            # cased text: a capitalised census surname right after a first name is the surname even when it is
            # also an ordinary word ("Jonah North", "Micah March") - unless it starts a date ("March 9")
            if cased and nxt in self.lex.surnames and toks[j + 1][0][:1].isupper() and not toks[j + 1][0].isupper() \
                    and (self.lex.medical_strength(nxt) < 2 or not (j + 2 < n and low[j + 2] in MEDICAL_HEADS)) \
                    and nxt not in WEEKDAYS and nxt not in PHRASE_BOUNDARY \
                    and not (j + 2 < n and re.match(r"\d", toks[j + 2][0])):
                return j + 1
            return j

        for i in range(n):
            w = low[i]
            # 1) title + name: "mr. smith", "dr. patel", "doctor jones"
            if w.rstrip(".") in TITLES:
                j = i + 1
                if j < n and low[j] == ".":
                    j += 1
                if word_at(j):
                    mode = "title" if w.rstrip(".") not in {"doctor", "doc", "nurse"} else "lenient"
                    if self.name_ok(toks[j][0], mode, cased):
                        end = extend_surname(j, "lenient")
                        role = "CLINICIAN" if w.rstrip(".") in CLINICIAN_TITLES else "TITLED"
                        emit(i, end, "title", 0.95, f"name preceded by title '{w}'", role, True)
                continue
            # 2) relation / pet + name: "my wife , jane", "her son tommy"
            if w in POSSESSIVES and i + 1 < n and low[i + 1] in RELATIONS | PET_WORDS:
                j, appos = i + 2, False
                if j < n and low[j] == ",":
                    j, appos = j + 1, True
                if word_at(j):
                    nxt = low[j + 1] if j + 1 < n else "."
                    # appositive ("my wife , jane ,") or a following verb/punctuation make the slot a name
                    # slot, so the looser title-strength test applies; otherwise require a name-dominant token
                    head_slot = appos or low[i + 1] in PET_WORDS or nxt in NAME_SLOT_FOLLOWERS or not word_at(j + 1)
                    # a drug brand in a relation slot followed by a verb is a person ("my friend valium
                    # recommended ..."): the syntactic slot outweighs the lexical drug evidence
                    drug_named = head_slot and nxt in NAME_SLOT_FOLLOWERS and self.lex.medical_strength(low[j]) == 2 \
                        and low[j] not in self.lex.clinical and low[j].isalpha()
                    if self.name_ok(toks[j][0], "title" if head_slot else "strict", cased) or drug_named:
                        role = "PET" if low[i + 1] in PET_WORDS else "FAMILY"
                        emit(j, extend_surname(j), "relation", 0.9, f"name introduced by '{w} {low[i + 1]}'", role, True)
                continue
            # 2b) cased text, relation noun directly followed by a capitalised name ("lives with niece Sunny")
            if cased and w in RELATIONS and word_at(i + 1) and toks[i + 1][0][:1].isupper() \
                    and self.name_ok(toks[i + 1][0], "title", cased):
                emit(i + 1, extend_surname(i + 1), "relation", 0.85, f"name after relation noun '{w}'", "FAMILY", True)
            # 3) greeting vocatives: "hi , andrew ." / "thank you , mary"
            g_end = None
            if w in GREETINGS:
                g_end = i
            elif w == "thank" and i + 1 < n and low[i + 1] == "you":
                g_end = i + 1
            elif w == "good" and i + 1 < n and low[i + 1] in {"morning", "afternoon", "evening", "night"}:
                g_end = i + 1
            elif w == "take" and i + 1 < n and low[i + 1] == "care":
                g_end = i + 1
            elif w == "see" and i + 1 < n and low[i + 1] == "you":
                g_end = i + 1 + (i + 2 < n and low[i + 2] in {"soon", "later", "then", "again"})
            if g_end is not None:
                j = g_end + 1
                if j < n and low[j] == ",":
                    j += 1
                if word_at(j):
                    # "hey hope , ..." - a lexicon name closed by punctuation is a vocative slot
                    closed = j + 1 >= n or not is_word[j + 1]
                    if self.name_ok(toks[j][0], "title" if closed else "strict", cased):
                        end = extend_surname(j)
                        if closed or end + 1 >= n or not is_word[end + 1] or self.name_ok(toks[j][0], "strict", cased):
                            emit(j, end, "greeting", 0.85, f"vocative after greeting '{w}'", "ADDRESSEE", False)
                continue
            # 3b) turn-initial vocative: "[doctor] andrew , how are you ?" / "[patient_guest] mary , ..."
            if not cased and i >= 1 and low[i - 1] == "]" and is_word[i] and i + 1 < n and low[i + 1] == "," \
                    and self.name_ok(toks[i][0], "lenient", cased) and self.lex.medical_strength(w) == 0:
                emit(i, i, "turn-vocative", 0.8, "name addressed at the start of a turn", "ADDRESSEE", False)
            # 4) comma vocative in dialogue: ", john ?"
            if w == "," and not cased and word_at(i + 1) and self.name_ok(toks[i + 1][0], "strict", cased):
                end = extend_surname(i + 1)
                if end + 1 < n and low[end + 1] in {"?", ".", "!", ","} and (end + 1 == n - 1 or low[end + 1] != "," or not word_at(end + 2) or low[end + 2] in {"how", "can", "i", "you", "we", "what", "do", "are", "is"}):
                    emit(i + 1, end, "vocative", 0.75, "vocative between punctuation", "ADDRESSEE", False)
            # 5) self-introduction / naming
            if w in {"named", "called"} and word_at(i + 1) and self.name_ok(toks[i + 1][0], "lenient", cased):
                # "it is called Eastbrook Craft Services", "a shop called ...": the name of a business
                end = i + 1
                while end + 1 < n and end - i < 5 and is_word[end + 1] and low[end + 1] not in PHRASE_BOUNDARY \
                        and (not cased or toks[end + 1][0][:1].isupper()):
                    end += 1
                subject = [low[k] for k in range(max(0, i - 3), i) if is_word[k] and low[k] not in {"is", "was", "the", "a", "an"}]
                if (subject and subject[-1] in NAMED_THING_SUBJECTS | {"it's", "that's"}) or low[end] in ORG_NAME_SUFFIXES:
                    head = low[end] if low[end] in INSTITUTION_HEADS else None
                    work = EMPLOY_RE.search(text[max(0, toks[i][1] - 200):toks[i][1]])
                    out.append(_sp(field, toks[i + 1][1], toks[end][2], "ORGANIZATION", "naming-org", text, 0.85,
                                   f"name of a business after '{w}'", strong=True,
                                   org_type=INSTITUTION_HEADS[head] if head else "employer" if work else None))
                    continue
            if w in {"named", "called"} or (w == "is" and i >= 1 and low[i - 1] in {"name"}) or (w == "name's"):
                j = i + 1
                if word_at(j) and self.name_ok(toks[j][0], "lenient", cased):
                    emit(j, extend_surname(j, "lenient"), "naming", 0.9, f"name after '{w}'", None, True)
            if w in {"i'm", "this"} or (w == "i" and i + 1 < n and low[i + 1] == "am"):
                j = i + (2 if w in {"this", "i"} else 1)
                if w == "this" and not (i + 1 < n and low[i + 1] == "is"):
                    continue
                if word_at(j) and self.name_ok(toks[j][0], "strict", cased):
                    end = extend_surname(j)
                    if end + 1 >= n or not is_word[end + 1] or low[end + 1] in {"and", "from", "with", "i'm"}:
                        emit(j, end, "self-intro", 0.8, "self-introduction", None, False)
            # 6) "seeing X" / "patient X" (dictation openers)
            if w in {"seeing", "patient"} and word_at(i + 1) and low[i + 1] not in {"is", "was", "today", "a", "an", "the"}:
                j = i + 1
                # "i'm seeing will kowalski today": a two-token name closed by today/punctuation is a
                # dictated patient name even if each token alone is ambiguous
                two = word_at(j + 1) and (j + 2 >= n or low[j + 2] in {"today", "this", ",", ".", "who", "for", "here"})
                if two and low[j] in self.lex.first_names and low[j] not in NOT_NAMES \
                        and self.name_ok(toks[j + 1][0], "title", cased) and low[j + 1] in self.lex.surnames \
                        and self.lex.zipf(low[j + 1]) < 4.5:
                    emit(j, j + 1, "dictation", 0.85, f"patient named after '{w}'", "PATIENT", True)
                elif self.name_ok(toks[j][0], "lenient", cased) and self.lex.name_ratio(low[j]) >= 0.3:
                    emit(j, extend_surname(j, "title"), "dictation", 0.85, f"patient named after '{w}'", "PATIENT", True)
            # 7) first-name + surname bigram
            if is_word[i] and word_at(i + 1) and low[i] in self.lex.first_names and low[i + 1] in self.lex.surnames:
                if self.name_ok(toks[i][0], "strict", cased) and self.name_ok(toks[i + 1][0], "lenient", cased) \
                        and self.lex.name_ratio(low[i + 1]) >= 0.3:
                    emit(i, i + 1, "full-name", 0.8, "first name followed by surname", None, False)

        # 8) names dictated right before DOB / MRN ("john smith , date of birth ...")
        for m in re.finditer(DOB_KEY + r"|\bmrn\b|\bmedical\s+record\s+number", text, I):
            k = next((x for x in range(n) if toks[x][1] >= m.start()), n) - 1
            while k >= 0 and low[k] in {",", ".", ":"}:
                k -= 1
            end, start = k, k
            while start >= 0 and k - start < 3 and is_word[start] and self.name_ok(toks[start][0], "title", cased):
                start -= 1
            if start < end:
                emit(start + 1, end, "dictation-dob", 0.9, "name dictated before date of birth / MRN", "PATIENT", True)
        # 8b) header fields of structured documents ("Patient: Doe, Jane", "Attending: Dr. Roe")
        for m in HEADER_NAME_RE.finditer(text):
            words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]+", m.group("name"))
            if words and not all(w.lower() in NOT_NAMES for w in words) and any(
                    self.lex.in_name_lexicon(w.lower()) or self.lex.zipf(w.lower()) < 3.5 for w in words):
                key = m.group("key").lower()
                role = "PATIENT" if key.split()[0] in {"patient", "name", "pt"} else \
                    "FAMILY" if key.split()[0] in {"emergency", "next", "guardian", "spouse", "contact"} else "CLINICIAN"
                out.append(_sp(field, m.start("name"), m.end("name"), "PERSON", "header-field", text, 0.95,
                               f"value of '{m.group('key')}:' header field", role_hint=role, strong=True,
                               name_tokens=[w.lower() for w in words]))
        # 8c) dictated subject in lowercase dialogue: "[doctor] andrew perez is a 62-year-old male ..."
        if not cased:
            for m in DIALOGUE_SUBJECT_RE.finditer(text):
                words = m.group("name").split()
                if all(self.name_ok(w, "lenient", cased) or (w in self.lex.surnames and self.lex.zipf(w) < 4.0)
                       for w in words) and any(self.name_ok(w, "lenient", cased) for w in words):
                    out.append(_sp(field, m.start("name"), m.end("name"), "PERSON", "dialogue-subject", text, 0.9,
                                   "subject of 'X is a N-year-old' dictation", role_hint="PATIENT", strong=True,
                                   name_tokens=words))
        # 9) cased note subject: "Andrew Perez is a 62-year-old ..."
        if cased:
            for m in NOTE_SUBJECT_RE.finditer(text):
                words = m.group("name").split()
                if all(w.lower() not in NOT_NAMES and w.lower() not in RELATIONS for w in words) and \
                        any(self.lex.in_name_lexicon(w.lower()) or self.lex.zipf(w.lower()) < 3.5 for w in words):
                    out.append(_sp(field, m.start("name"), m.end("name"), "PERSON", "note-subject", text, 0.95,
                                   "subject of 'X is a N-year-old' sentence", role_hint="PATIENT", strong=True,
                                   name_tokens=[w.lower() for w in words]))
        return out

    # ------------------------------------------------------------------ places
    def _places(self, text: str, field: str, toks: list, cased: bool) -> list[Span]:
        out = []
        # named natural landmarks ("mount hood", "Lake Tahoe", "Zion National Park") give a masked state back;
        # the name must be a rare word and not a clinical criterion ("Lake Louise score")
        for m in _LANDMARK_RE.finditer(text):
            name = (m.group("n") or m.group("n2")).lower()
            words = name.split()
            if any(self.lex.zipf(w) >= 4.5 or self.lex.medical_strength(w) >= 1 for w in words if w not in ("national", "park", "forest")) \
                    or re.match(r"\s*(?:criteria|score|scale|classification|questionnaire)\b", text[m.end():m.end() + 16], I):
                continue
            out.append(_sp(field, m.start(), m.end(), "LOCATION", "landmark", text, 0.8, "named natural landmark",
                           place_type="landmark"))
        for m in self._place_re.finditer(text):
            p = m.group("p").lower()
            surface = m.group("p")
            if cased and not surface[:1].isupper():
                continue
            before = re.findall(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ']+", text[max(0, m.start() - 25):m.start()].lower())[-2:]
            after = re.findall(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ']+", text[m.end():m.end() + 15].lower())[:1]
            cue = any(b in LOC_CUES for b in before) or (after and after[0] in {"state", "city", "county", "area"})
            first = p.split()[0]
            ambiguous = self.lex.name_ratio(first) >= 0.05 or self.lex.zipf(p.replace(" ", "")) >= 4.3 or \
                (len(p.split()) == 1 and self.lex.zipf(p) >= 4.0)
            if (cue or not ambiguous) and not (after and after[0] in {"sandwich", "dinner", "bacon", "meat", "virus", "fever", "disease"}):
                out.append(_sp(field, m.start(), m.end(), "LOCATION", "gazetteer", text, 0.85 if cue else 0.7,
                               f"gazetteer {self.lex.places[p]}" + (" with locative cue" if cue else ""),
                               place_type=self.lex.places[p], strong=bool(cue)))
        return out

    def _distinctive(self, tok: str, cased: bool) -> bool:
        tl = tok.lower()
        # medical/anatomical words make a *specialty* clinic ("Hand Clinic", "Heart Center"), not a name
        if tl in NOT_NAMES or tl in GENERIC_WORKPLACE or self.lex.medical_strength(tl) >= 1:
            return False
        if cased:  # title-case words are names; ALL-CAPS words in notes are section headers
            return tok[:1].isupper() and not (tok.isupper() and len(tok) > 1)
        return self.lex.zipf(tl) < 4.5 or self.lex.name_ratio(tl) >= 0.1 or tl in self.lex.places

    def _institutions(self, text: str, field: str, toks: list, cased: bool) -> list[Span]:
        out = []
        starts = {t[1]: k for k, t in enumerate(toks)}
        for m in _INST_RE.finditer(text):
            k = starts.get(m.start())
            if k is None or (cased and not m.group(0)[:1].isupper()):
                continue  # in cased text a named institution has a capitalised head ("Mercy Hospital", not "urgent care")
            # name words before the head ("ohio state UNIVERSITY", "st. luke's HOSPITAL"), up to 3,
            # stopping at function words; accepted if at least one of them is distinctive
            j, first, distinctive = k - 1, None, False
            while j >= 0 and k - j <= 4:
                t = toks[j][0]
                tl = t.lower()
                if "\n" in text[toks[j][2]:toks[j + 1][1]]:
                    break  # an institution name never spans a line break (section headers)
                if cased and re.match(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ]", t):
                    break  # in cased text the name of an institution is capitalised
                if cased and (j == 0 or toks[j - 1][0] in {".", ":", "?", "!", ";", "]"} or "\n" in text[toks[j - 1][2]:toks[j][1]]) \
                        and self.lex.zipf(tl) >= 4.5 and self.lex.name_ratio(tl) < 0.1 and tl not in self.lex.places:
                    break  # "Seek Urgent Care ...": a very common word capitalised only because it starts the sentence
                if tl in {"st", "st.", "saint", "."} or (re.match(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ]", tl) and tl not in PHRASE_BOUNDARY
                                                         and tl not in INSTITUTION_STOP):
                    distinctive |= tl not in {"st", "st.", "saint", "."} and self._distinctive(t, cased)
                    first = j
                    j -= 1
                    continue
                break
            if first is not None and toks[first][0] == ".":
                first += 1
            if first is not None and first < k and distinctive:
                kind = INSTITUTION_HEADS[m.group(0).lower()]
                label = "LOCATION" if kind == "location" else "ORGANIZATION"
                out.append(_sp(field, toks[first][1], m.end(), label, "institution", text, 0.85,
                               f"named {m.group(0).lower()}", org_type=kind, strong=True))
        for m in EMPLOY_RE.finditer(text):
            k = starts.get(m.end())
            if k is None:
                continue
            # employer phrase = words up to a boundary (punctuation / function word), max 4 tokens;
            # accepted when at least one word is distinctive (rare, name-like or capitalised)
            j, distinctive = k, False
            # "works at a pet-supply store" describes a kind of workplace; a name is capitalised in cased
            # text and, after an indefinite article, needs name evidence even in lower-case dialogue
            generic_np = k > 0 and toks[k - 1][0].lower() in {"a", "an"}
            while j < len(toks) and j - k < 4:
                t = toks[j][0]
                tl = t.lower()
                if not re.match(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ]", tl) or tl in PHRASE_BOUNDARY or self.lex.medical_strength(tl) == 2:
                    break
                rare = self.lex.zipf(tl) < 4.5 and not ("-" in tl and all(self.lex.zipf(p) >= 4.0 for p in tl.split("-") if p))
                if tl not in GENERIC_WORKPLACE and tl not in NOT_NAMES and self.lex.medical_strength(tl) < 2 and (
                        self.lex.name_ratio(tl) >= 0.1 or (cased and t[:1].isupper()) or (rare and not cased and not generic_np)):
                    distinctive = True
                j += 1
            if j > k and distinctive:
                out.append(_sp(field, toks[k][1], toks[j - 1][2], "ORGANIZATION", "employer", text, 0.75,
                               "named employer / workplace", org_type="employer", strong=True))
            elif j == k + 1 and toks[k][0].lower() not in GENERIC_WORKPLACE | NOT_NAMES and \
                    self.lex.medical_strength(toks[k][0].lower()) < 2 and (j >= len(toks) or not re.match(r"\w", toks[j][0])):
                # a single ordinary word in the employer slot ("work at shell .") is still a company name
                out.append(_sp(field, toks[k][1], toks[k][2], "ORGANIZATION", "employer", text, 0.65,
                               "single word in an employer slot", org_type="employer", strong=True))
        return out

    # ------------------------------------------------------------------ occupations
    def _occupations(self, text: str, field: str) -> list[Span]:
        out = []
        for m in self._occ_re.finditer(text):
            left = text[max(0, m.start() - 60):m.start()]
            if not OCC_CUE_RE.search(left):
                continue
            # extend over up to 3 modifiers between the article/cue and the occupation head
            mods = re.findall(r"([A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ\-']+)\s+$", left)
            start = m.start()
            pre = re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ\-']+|,", left)
            taken = 0
            for tok in reversed(pre):
                if tok.lower() in OCC_STOP or tok.lower() in self.lex.occupations or taken >= 3:
                    break
                start = text.rfind(tok, 0, start)
                taken += 1
            occ = m.group("occ").lower()
            out.append(_sp(field, start, m.end(), "OCCUPATION", "occupation", text, 0.8,
                           "occupation stated for a person", category=self.lex.occupations.get(occ, "other"),
                           modifiers=bool(mods)))
        # lexicon-free: the noun phrase after an explicit job cue ("works as a crane operator at ...")
        for m in OCC_PHRASE_RE.finditer(text):
            keep = []
            for w in m.group("occ").split():
                if w.lower() in PHRASE_BOUNDARY or w.lower() in OCC_STOP or w.lower() in {"at", "for", "with", "and", "but"} or self.lex.medical_strength(w.lower()) == 2:
                    break
                keep.append(w)
            if not keep or " ".join(keep).lower() in {"well", "usual", "normal", "always", "much", "long"}:
                continue
            s = m.start("occ")
            e = s + len(" ".join(keep))
            head = keep[-1].lower()
            cat = self.lex.occupations.get(" ".join(keep).lower()) or self.lex.occupations.get(head) or \
                self.lex.occupations.get(head.rstrip("s"), "other")
            out.append(_sp(field, s, e, "OCCUPATION", "occupation-cue", text, 0.75,
                           "job title following an explicit occupation cue", category=cat))
        return out

    # ------------------------------------------------------------------ lab values (for perturbation)
    def _labs(self, text: str, field: str) -> list[Span]:
        out = []
        for m in _LAB_RE.finditer(text):
            lab = m.lastgroup
            window_end = min(len(text), m.end() + 70)
            stop = re.search(r"[?!\n]|\.(?!\d)\s|\[", text[m.end():window_end])
            window_end = m.end() + stop.start() if stop else window_end
            nxt = _LAB_RE.search(text, m.end(), window_end)
            if nxt:
                window_end = nxt.start()
            prev_end = None
            for v in _LAB_VALUE_RE.finditer(text, m.end(), window_end):
                if prev_end is not None and not _TREND_CONNECTOR.search(text[prev_end:v.start()]):
                    break  # after the first result, only values of a trend of the same test ("from 9.6%")
                raw = v.group("v")
                before = text[max(m.end(), v.start() - 8):v.start()].lower()
                ve = v.end("v")  # end of the number itself (the whole match also swallows the next token)
                if _TIME_UNIT_AFTER.match(text[ve:ve + 12]) or re.search(r"\b(?:in|every|x|for)\s*$", before):
                    continue
                if re.match(r"\s*(?:-|–|to)\s*\d", text[ve:ve + 6]) or re.search(r"\d\s*(?:-|–|to)\s*$", before) or (
                        re.search(r"\bbetween\s*$", text[max(m.end(), v.start() - 10):v.start()], I)
                        and re.match(r"\s*and\s*\d", text[ve:ve + 8], I)):
                    break  # part of a reported range ("EF 55-60%"): perturbing one end would distort it
                val = words_to_number(raw)
                lo, hi = LABS[lab][2]
                if val is None or not (lo <= val <= hi):
                    break
                decimals = len(raw.split(".")[1]) if "." in raw else (
                    len(raw.lower().split("point")[1].split()) if "point" in raw.lower() else 0)
                out.append(_sp(field, v.start("v"), v.end("v"), "LAB_VALUE", "lab", text, 0.8,
                               f"{lab} result", lab=lab, value=val, decimals=decimals,
                               form="digits" if raw[0].isdigit() else "words"))
                prev_end = ve
        return out
