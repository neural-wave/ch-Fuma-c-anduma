# MedAnon – Round 6 DEV48 (synthetic development data)

48 synthetic DEV encounters (offsets of all 3,092 reference spans validated). Round 6 validation inputs/gold and
Round 5 were not opened. Baseline B3 = `18fdb0e` (code `6da2d316`); candidate B4 = this commit (code `b7d44883`).
Both run on the same inputs, GPU (`--device cuda`), default and `configs/age_bands.yaml`; outputs saved per version
under `run5/outputs_dev48/`. Peak VRAM 2,619 MiB, ~0.70 s/encounter, structure intact 48/48.

## Core privacy (labels PERSON, AGE, DOB, DATE, ID, CONTACT, ADDRESS, LOCATION, ORGANIZATION, OCCUPATION)

| | B3 | B3+age | B4 | B4+age |
|---|---|---|---|---|
| Raw recall | 0.919 | 0.990 | 0.919 | 0.990 |
| Policy-compliant recall | 1.000 | 1.000 | 1.000 | 1.000 |
| Precision vs all non-clinical gold* | 0.990 | – | – | 0.998 |
| Strict token removal (proxy) | 0.715 | 0.840 | 0.714 | 0.840 |
| Partial identifiers / records with a direct identifier left | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Clinical spans wrongly replaced (of 1,180) | 14 | 14 | **3** | **3** |
| Records below k=3 | 45 | 21 | 45 | 21 |

*The scorer's core precision (0.76–0.88) counts correct ANTHROPOMETRIC / FAMILY_SIZE transformations as false
positives because those labels are outside the core denominator; counted against all non-clinical gold it is
0.990 (B3+age) → 0.998 (B4+age).

## Policy-sensitive objectives (share of reference spans transformed)

| | B3 | B4 |
|---|---|---|
| Height, demographic | 28/72 | 60/72 |
| Weight, demographic | 28/124 | 120/124 |
| BMI, demographic | 56/62 | 60/62 |
| Height / weight / BMI marked clinically sensitive (should stay exact) | 10/40 · 10/68 · 20/34 | 20/40 · 40/68 · 20/34 |
| Family size (58 counts ≥ 3; 0–2 kept by policy) | 0/96 | 58/96 |
| Family relation (son/daughter; not transformed by design) | 0/126 | 0/126 |

The reference "clinically sensitive" flag could not be derived from the text: the only explicit cue
("implications for medication dosing") appears in all 48 records, and similar diagnoses carry opposite flags.
Exemptions therefore follow explicit clinical rules (children, weight-based dosing, dialysis/fluid, gestational
weight gain, malnutrition, Z68 codes) instead of the generator labels; more detected measurements means more of
these flagged values are generalised (to 5 cm / 5 kg / WHO-class intervals, never removed).

Rounds 2/3/4 (non-regression, same metric definitions): identical recall; precision R3 0.990 → 0.997; clinical
spans wrongly replaced R3 1 → 0; no new partial identifiers.

## Changes (each with positive tests and counterexamples, `tests/test_focused_quasi.py`, 22 cases)

Family size from household statements ("there are three children", "household composition: 3 children",
siblings), counts 0–2 kept; heights/weights from height-weight pairs, unit restatements and "on the scale";
pregnancy exemption limited to gestational weight-gain monitoring; unit glued to the next sentence; mixed-case
analytes (tTG-IgA, IgA); hyphenated instruments (DXA T-score) and eponyms (Dix-Hallpike).

## Open

Clinically sensitive measurements generalised per the reference flag (see above); occupation partials by
reference convention (status words kept); ID cue-word convention in Round 3; Bosch's 0.5918 not reproducible locally.
