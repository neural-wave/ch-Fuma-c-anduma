# MedAnon re-identification benchmark (research module)

An independent, synthetic, adversarial evaluation of whether patients can be re-identified from MedAnon output
using plausible auxiliary information. It does not modify MedAnon: it treats MedAnon as a black box (CLI run on
identical inputs per policy) and reads only its output files. Results: [`REPORT.md`](REPORT.md).

**Fictional data only.** Every person, name, place and record is generated (Synthea + the generator here).
No Bosch data, no official evaluation data and no validation gold are read by any part of this module.

## Pipeline

```
Synthea v4.0.0 (Apache-2.0, Java 21)  ->  reid/world.py  ->  world dir
   population of ~11,000 fictional          persons + encounters + registers
   people, Massachusetts geography          (private/, release_input/, attacker/)
                                                     |
               MedAnon (ceff209) CLI, one run per policy, same input file
                                                     v
                                     anonymized/<policy>/encounters.jsonl
                                                     |
   reid/attack/run.py  (separate process; sees ONLY the anonymised file + one register)
        extract.py   features from the published text
        compare.py   comparison levels (numpy) == Splink SQL levels (checked pair by pair)
        Splink 5.0.0 Fellegi-Sunter: u by random sampling, m by EM (no labels)
                                                     v
                                     attack/<policy>/<threat>/scores.npz
                                                     |
   reid/evaluate.py + reid/clinical.py  (sees truth)  ->  results/*.json  ->  REPORT.md
```

## Data visibility

| Directory | Contents | Who reads it |
|---|---|---|
| `private/` | person truth (whole population), encounter -> person, register key -> person, clinical facts, generator manifest | evaluator only |
| `release_input/` | identifiable MedAnon input | MedAnon only; the evaluator for clinical checks |
| `anonymized/<policy>/` | MedAnon output | attacker |
| `attacker/aux_{A,B,C}.parquet` | one register per threat model, whole population, random opaque keys | attacker |

The attacker CLI refuses paths under `private/` or `release_input/`; a test checks that `reid/attack/` imports no
generator, truth or evaluator code. Encounter ids and register keys are independent random draws; files are shuffled.
Every person, target or not, gets a register row from the same noisy process, so presence is no shortcut.

## Population and encounters

- Synthea supplies sex, birth date, race/ethnicity, marital status, town, income, employment and education findings,
  conditions with onset dates, surgeries, medications, vitals and labs. Names lose Synthea's digit suffixes.
- The generator adds, with stated correlations: occupation (from Synthea employment and education findings, and
  age), work pattern (shift work, remote work, long-haul driving, part-time), number of children (by age and
  marital status, at least the number of Synthea births for women), a hobby, and visible life events
  (fractures, concussions, heart attacks, childbirths, surgeries).
- About 300 encounters per world come from roughly 210-230 patients. Most patients have one encounter; some have
  two or three, ordered in time. Each encounter is drawn from the patient's real Synthea encounters in 2023-2026.
  Problem visits are weighted three times more than check-ups.
- The text follows the MedAnon schema (`encounter_id, dataset, dialogue, note, icd10_codes`). Dialogues are
  ASR-like: lowercase, with fillers and number words. Notes are cased and sectioned. The text includes uncertain
  dates ("2014 maybe 2015"), relative durations, family history with relatives' ages, and a dictated recap. Each
  quasi-identifier appears only with a set probability.
- **Hard cases.** For about 12 % of target patients, the population contains two doppelgangers. They share sex,
  birth year, town, race, marital status, children, occupation and chronic-condition set, but have different names
  and onset years shifted by 0-2. They are always placed in the target's candidate pool.
- **Splits.** Dev worlds `dev1` and `dev2` come from Synthea population seed 101 and use the dev template family.
  The blind world `test1` comes from an independent population (seed 202) and uses the test template family.
  Every phrasing list is split by index parity, so test phrasings never occur in dev.

## Threat models (auxiliary knowledge; exact rates in `reid/world.py: THREATS`)

Every attacker knows the target's name, because the attacker is looking for that person. Every attacker also knows
the MedAnon release format and its public lexicons (Kerckhoffs). Each piece of knowledge has a stated availability
and error rate, and some values are stale or wrong.

| | A - public information | B - acquaintance | C - informed |
|---|---|---|---|
| Example | social media and professional profile | coworker, neighbour | insurer, former caregiver, or someone with partial records |
| Sex | 98 % | 100 % | 100 % |
| Birth | birth year 45 %, age ±3 35 % | full date 50 %, year 35 %, age ±2 15 % | full date 85 %, year 13 % |
| Town | 65 % (12 % stale) | 90 % (5 % stale) | 95 % |
| Occupation title | 55 % (10 % stale) | 85 % (5 % stale) | 90 % |
| Race (apparent) | 70 % (5 % wrong) | 95 % | 97 % |
| Marital status | 30 % | 95 % | 98 % |
| Children, hobby, work pattern | - | 90 % (12 % off by one), 75 %, 70 % | 97 %, 80 %, 80 % |
| Height/weight | - | impression: ±8 cm / ±12 kg interval, 60 % | measured: ±1 cm / ±2 kg, 80 % |
| Life events (year) | - | each 65 %, year ±1 in 25 % | each 85 % |
| Chronic diagnoses (ICD-10, 3 chars) | - | - | each 60 %, onset year 70 % |
| A clinic visit (year-month) | - | - | 50 %; any visit in the window, released or not |

The public-information attacker never gets medical or body information. The acquaintance gets only life events
that people notice. Only attacker C has diagnoses, and they are partial.

## Attackers and scoring

- **Feature extraction** reads only the published text. An age band stays an interval. An age at an unknown visit
  year becomes a birth-year interval over the release window, 2023-2026, which the attacker is assumed to know. A
  calendar year counts as a birth year only when it is published as one (`[DOB] 1978`). Visit-header years and
  clinical-event years are kept separate. Residual names, towns, occupation titles and hobbies are found by
  dictionary search over the attacker's own register. A value that is not published is "no information", never a
  disagreement.
- **Deterministic baseline.** A candidate is compatible when no published attribute contradicts it. The attacker
  identifies only a unique compatible candidate with at least 2 agreeing attributes, and abstains otherwise.
- **Probabilistic attacker** (Splink 5.0.0, DuckDB). Fellegi-Sunter with interval-aware custom comparison levels.
  u comes from random sampling and m from EM over two blocking passes; no labels are used. A level that no EM
  session observes is set to "no evidence" (m = u) and logged. The decision uses a pool posterior,
  `BF_j / ((1-pi)/pi * P + sum_k BF_k)`. The attacker identifies a candidate when the posterior reaches 0.9, a
  threshold fixed a priori. pi is 1 in the closed world and 0.5 in the open world, as the attacker's assumption.
- **Random baseline**, analytic: top-1 = 1/P.
- **Pools.** P = 100, 500, 2,000 and 10,000 per target patient. Decoys are drawn from the population with seeds set
  by world, P and patient, so they are identical across policies and threat models. Doppelgangers are forced in.
  In the open world each patient is present with probability 0.5.
- **Metrics.** Top-1, top-5 and MRR. Success over all eligible targets, precision among identifications, coverage,
  false-identification rate, and false-match rate when the target is absent (overall and for hard cases).
  Patient-level re-identification. Identity-to-record inference of the primary-diagnosis ICD-10 chapter, against
  prior-only and random-record baselines. 95 % percentile bootstrap over patients (B = 1000); patients shared
  across dev worlds form one cluster. Paired policy differences use the same queries and pools.
- **Controls.** Shuffled record-to-patient mapping, a register with independently permuted columns, a register
  with half of the knowledge deleted, a register of an independent population (every target absent), and a
  demographics-only attacker.

## Run

```bash
PY=D:/NEURAL_WAVE/.venv-reid/Scripts/python        # Python 3.12, splink==5.0.0, duckdb 1.5.6, pandas, pyarrow
cd research/reid
# 1. population (Java 21): java -jar synthea-v4.0.0.jar -p 11000 -s 101 -cs 101 -r 20261001 --exporter.csv.export=true \
#    --generate.only_alive_patients=true --exporter.years_of_history=8 --exporter.baseDirectory=./out Massachusetts
$PY -m reid.world --synthea <pop_101>/out/csv --out <runs>/dev1 --seed 1 --family dev
# 2. MedAnon on <runs>/dev1/release_input/encounters.jsonl, once per policy -> <runs>/dev1/anonymized/<policy>/encounters.jsonl
# 3. attack + evaluate
$PY -m reid.bench attack   --runs <runs> --worlds dev1 dev2 --policies original default age_bands age_bands_strict_dates
$PY -m reid.bench controls --runs <runs> --world dev1 --policy default --indep-register test1
$PY -m reid.bench evaluate --runs <runs> --worlds dev1 dev2 --pooled --out results/dev.json
$PY -m pytest tests -q
```
