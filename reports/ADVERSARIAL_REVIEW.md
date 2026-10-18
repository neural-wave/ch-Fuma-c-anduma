# Review of the teammate adversarial report + temporal-policy check (synthetic development data)

## The teammate report (100 synthetic adults, linkage attacker)
Ran on commit `096ff01`, rules-only detectors. Its "adult_age_bands" row is the legacy override
`AGE: generalize, band_width 10`, which keeps the birth year — the report itself notes that a retained DOB
year narrows age to two values. It therefore does not test `configs/age_bands.yaml` (adult bands + adult
birth year removed), which did not exist at `096ff01`. Its strongest effect, DATE year removal (88% -> 43%
unique correct), depends on an auxiliary "administrative registration year" that varies across its 100
people; closed world, complete and accurate knowledge -> an upper-bound stress test. The ICD proxy has a
label space of 4 (F1 1.0 everywhere), so it cannot show clinical damage; the clinical screen blocks every
policy because of an occupation loss already present at baseline. Reproducing it on the current code needs
the attack script, its JSON output, the 100-person dataset (sha256 f714303d...) and the auxiliary table.

## Same mechanism on our development sets (`scripts/linkage_check.py`, code dd32164 = b7d44883)
Attacker knows exact age, sex, encounter year; reads only the anonymised text (exact age, band, surviving
DOB year, `[DATE] YYYY`). Targets uniquely and correctly linked:

| Set | default (exact age) | age_bands | age_bands + no date year |
|---|---|---|---|
| Round 6 DEV48 (48) | 33 | 6 | 6 |
| Round 4 DEV20 (20) | 20 | 4 | 4 |

The remaining links are children (exact age kept by policy). Removing date years gives no further linkage
benefit here because almost all encounter dates fall in one year; it raises strict token removal
(0.840 -> 0.877 R6, 0.762 -> 0.794 R3, 0.732 -> 0.799 R4) with no clinical span changed. Rounds 2/3 gold
ages are partly spelled out, so the attacker has no age knowledge there (not reported).

## Decision
`configs/age_bands.yaml` remains the recommended profile. `configs/age_bands_strict_dates.yaml` is provided
as an experimental option: useful where encounter years vary across patients; its cost (year of a full-date
clinical event is lost; isolated years and relative dates kept) is not measurable on these sets.
