"""Synthetic probe for DOB / age / year / person-placeholder behaviour at DATASET level (k-anonymity on).
All content is invented. Prints each probe line before -> after."""
import sys

from medanon import Anonymizer

PROBES = [
    ("P01", "[doctor] hi tomasz , what's your date of birth ?\n[patient] march twenty second , nineteen eighty five .\n",
     "Tomasz Wierzbicki is a 40-year-old male. DOB: 14/07/2013. Born on March 22, 1985.\n"),
    ("P02", "[doctor] how old is he ?\n[patient_guest] he's six .\n",
     "Pediatric visit. The patient is a 6-year-old boy, DOB 2019-03-04. Born in 2019.\n"),
    ("P03", "[doctor] okay mister quill you're ninety two ?\n[patient] yes ninety two years old .\n",
     "Mr. Quill is a 92-year-old male. His mother had a stroke at age 70.\n"),
    ("P04", "[patient] i had my knee surgery in 2018 with doctor jonathan smith and then in twenty twenty one again .\n",
     "PAST SURGICAL HISTORY\n\nKnee arthroscopy in 2018 with Dr. Jonathan Smith. Appendectomy in March 2015. "
     "Diagnosed with diabetes in 2009.\n"),
    ("P05", "[patient] my pin is 2018 and i live at 2018 elm street .\n",
     "MRN: 20184411. Address: 2018 Elm Street. The patient is a 37-year-old woman. Hypertension since 2011.\n"),
    ("P06", "[doctor] hi linnea , this is my colleague doctor brandt and your husband osvaldo is here .\n"
            "[patient] yes , and my sister ingrid called .\n",
     "Linnea Halvorsen is a 65-year-old female seen with Dr. Brandt. Her husband Osvaldo and sister Ingrid are supportive.\n"),
    ("P07", "[doctor] hello there .\n[patient] hi .\n", "The patient is an 89-year-old male with COPD since 2016.\n"),
    ("P08", "[doctor] hello there .\n[patient] hi .\n", "The patient is a 37-year-old female. Hypertension diagnosed in 2020.\n"),
]


def main():
    anon = Anonymizer()
    recs = [{"encounter_id": e, "dataset": "probe", "dialogue": d, "note": n,
             "icd10_codes": [{"code": "I10", "description": "Essential (primary) hypertension"}]} for e, d, n in PROBES]
    results, stats = anon.anonymize_dataset(recs)
    for r, (o, _) in zip(recs, results):
        print(f"===== {r['encounter_id']}")
        for f in ("dialogue", "note"):
            for a, b in zip(r[f].split("\n"), o[f].split("\n")):
                if a.strip():
                    print(f"  {a}\n  -> {b}")
    print("escalation:", stats.get("escalation_levels"), "| qi:", {k: v for k, v in stats.get("qi_verification", {}).items() if k != "scope"})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
