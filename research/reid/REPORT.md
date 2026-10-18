# MedAnon re-identification benchmark: results (synthetic, 18 Oct 2026)

**Scope.** These are attack results on fictional people and records (Synthea plus generated text), against
MedAnon commit `ceff209` run with its production detectors on GPU. **No percentage below is a real-world
re-identification probability, a Bosch privacy score, or a HIPAA/GDPR statement.** The numbers describe
this generator, these threat models and this attacker. Design and data visibility: [README.md](README.md).

## Setup actually run

| Item | Value |
|---|---|
| MedAnon | `ceff209` (git archive), code sha256 `ac834ee0…`. Production detectors: rules + spaCy lg + dslim BERT + OpenMed 434M, `--offline --device cuda`. One run per policy on the identical input file. |
| Policies (config sha256) | `default` (`ad22a538…`); `configs/age_bands.yaml` (`dedfed9e…`); `configs/age_bands_strict_dates.yaml` (`f3352a54…`). Reference condition `original` = no anonymisation. |
| MedAnon cost | RTX PRO 5000 (48 GB): 9 runs × 300 encounters, 75–79 s per run (≈0.26 s/encounter including model load), peak VRAM 2,639 MiB, peak RSS ≈3.6 GB (`results/gpu/`). |
| Population | Synthea v4.0.0 (Apache-2.0, JRE 21), Massachusetts geography. Dev population seed 101 (11,000 people, 7.5 min). Blind population seed 202 (11,000, 7 min). Each world adds 50–54 doppelgangers. |
| Worlds | `dev1` (seed 1): 300 encounters, 228 patients. `dev2` (seed 2): 300 / 209. Blind `test1` (seed 3, independent population, disjoint template family): 300 / 224. About 12 % of targets are hard cases with doppelgangers. |
| Attacker | Splink 5.0.0 (DuckDB 1.5.6), Python 3.12. Per (world, policy, threat): 3.3 M pairs; extraction + EM + scoring 65–90 s on a laptop CPU. The numpy and Splink comparison codings agree on 100 % of pairs in every run. |
| Freeze | Attacker and evaluator hashes recorded before the single blind run (`results/frozen_attacker.sha256`, re-verified OK). |
| Tests | `pytest research/reid/tests`: 15 passed. All are hand-computable fixtures for levels, pool posterior, metrics, open-world denominators, parsing of every MedAnon output form, and attacker isolation. |

The dev worlds were used for attacker development, including two extractor fixes motivated by MedAnon dev output
and four bug fixes found by tests and controls. The blind world was evaluated once after the freeze.

## 1. Main results, candidate pool P = 2,000 (FS = probabilistic attacker, threshold 0.9 fixed a priori)

Percentages are encounter-level with a 95 % patient-cluster bootstrap. "Success" counts correct identifications
over all eligible targets. In the open world, half of the patients are absent from the pool.

**Dev (dev1 + dev2 pooled, 600 encounters, 437 patient clusters)**

| Policy | Thr. | Top-1 | Success (closed) | Precision | Success (open) | False match, absent target | Det. baseline success / precision | Patients re-identified |
|---|---|---|---|---|---|---|---|---|
| original | A | 87.7 (85–90) | 78.7 (75–82) | 98.1 | 63.7 | 1.1 | 57.8 / 98.6 | 84.2 |
| original | C | 98.8 (98–100) | 95.8 (94–97) | 99.8 | 92.9 | 3.1 | 57.5 / 98.9 | 97.7 |
| default | A | 2.5 (1–4) | 0.5 (0–1) | 50 (n=2) | 0.0 | 0.0 | 0.3 / 67 | 0.7 |
| default | B | 9.0 (7–11) | 0.8 (0–2) | 100 | 0.0 | 0.0 | 6.7 / 76 | 1.2 |
| default | C | 61.3 (57–66) | **38.2 (34–43)** | 92.7 | 31.3 | 4.2 | 28.3 / 82 | 43.7 |
| age_bands | A | 3.2 | 0.2 | – | 0.0 | 0.4 | 0.3 / 67 | 0.2 |
| age_bands | B | 10.2 | 0.3 | – | 0.3 | 0.0 | 4.2 / 83 | 0.5 |
| age_bands | C | 51.5 (47–57) | **21.5 (18–25)** | 94.2 | 15.0 | 0.4 | 19.8 / 79 | 24.6 |
| strict_dates | A | 3.2 | 0.3 | – | 0.0 | 0.0 | 0.3 / 67 | 0.5 |
| strict_dates | B | 10.3 | 0.3 | – | 0.3 | 0.0 | 3.2 / 86 | 0.5 |
| strict_dates | C | 49.8 (45–55) | **23.0 (19–27)** | 93.9 | 17.1 | 1.1 | 16.0 / 84 | 26.5 |

**Blind test (`test1`, 300 encounters, 224 patients, independent population, unseen phrasings)**

| Policy | Thr. | Top-1 | Success (closed) | Precision | Success (open) | False match, absent target | Det. success / precision |
|---|---|---|---|---|---|---|---|
| original | C | 98.0 | 96.3 | 100 | 95.7 | 3.1 | 53.0 / 99.4 |
| default | A | 1.7 | 0.0 | n/a | 0.0 | 0.0 | 0.0 |
| default | B | 29.3 (24–35) | **12.7 (9–17)** | **63.3** | 5.7 | 3.8 | 6.3 / 63 |
| default | C | 58.0 (52–64) | **34.3 (28–41)** | 98.1 | 32.1 | 4.4 | 27.7 / 83 |
| age_bands | B | 19.0 | 4.3 (2–7) | 59.1 | 0.7 | 0.0 | 1.7 / 63 |
| age_bands | C | 48.3 | 20.3 (16–25) | 83.6 | 14.3 | 3.1 | 15.7 / 75 |
| strict_dates | B | 10.3 | 1.3 (0–3) | 100 | 0.0 | 0.0 | 1.7 / 63 |
| strict_dates | C | 46.0 | 17.7 (13–22) | 93.0 | 10.7 | 3.1 | 12.0 / 84 |

Random baseline at P = 2,000: top-1 0.05 %, top-5 0.25 %.

**Pool size** (FS closed top-1 / success, dev pooled). default C: P = 100 79.7/71.2, P = 500 73.0/53.3,
P = 2,000 61.3/38.2, P = 10,000 47.3/21.5. age_bands C: 74.2/59.0, 62.5/37.3, 51.5/21.5, 37.8/8.2. Under threat A
the attacker never identifies confidently at P ≥ 500 for any anonymised policy. Hard cases (doppelgangers in the
pool), default C: success 28 % against 38 % overall. False matches on absent hard targets: ≤ 5 %.

**Paired policy differences** (same queries and pools, FS success, closed, P = 2,000)

| Comparison | Dev C | Blind C | Dev B | Blind B |
|---|---|---|---|---|
| default − age_bands | **+16.7 pp** (13.3–19.9) | **+14.0** (10.3–18.1) | +0.5 (0–1.2) | **+8.3** (4.9–12.1) |
| age_bands − strict_dates | −1.5 (−3.8 to +0.9) | +2.7 (−0.3 to +5.4) | 0.0 | +3.0 (1.3–4.9) |
| original − default | +57.7 | +62.0 | +91.2 | +74.7 |

**Attribute inference** (identity → record, primary-diagnosis ICD-10 chapter). Threats A and B are at the
prior-only baseline: dev 0.46–0.64 against a prior of 0.54–0.61. On blind `default`, B reaches 0.80 against a prior
of 0.64. Threat C reaches 0.81–0.94, but C already knows part of the diagnoses. A knowledge-only baseline was not
computed, so this value is an **upper bound, not a measured disclosure**.

## 2. Controls (dev1, default)

| Control | Expected | Observed (P = 2,000 unless noted) |
|---|---|---|
| Shuffled record → patient mapping | chance level | top-1 0.3–0.7 %, success 0–0.7 % |
| Register with independently permuted columns | chance level | top-1 ≤ 1.3 % at P = 100, success 0 |
| Register of an independent population (all absent) | no identifications | FS false-match rate 0–0.3 % |
| Half of the attacker's knowledge deleted | large drop | C success 38 % → 3.7 %, top-1 61 % → 20 % |
| Demographics only (sex + birth year) | low | success 0 % at P = 2,000; top-1 34–52 % at P = 100 |

There is no evidence of a generator shortcut. Success depends strongly on how complete the attacker's knowledge is.

## 3. What the results establish (and what they do not)

1. **Direct identifiers are almost entirely removed.** On dev, exact first or last names survive in 6/318 and
   6/512 encounters where they occurred. Full birth dates survive in 0/243, towns in 13/207 and exact occupation
   titles in 4/313. Without anonymisation, the same attacker re-identifies 78–96 % of encounters at P = 2,000.
2. **A public-information attacker (A) fails under all three policies.** Success is ≤ 0.5 % and never confident
   at P ≥ 500.
3. **An acquaintance (B) is phrasing-dependent.** B is near zero on dev and reaches 12.7 % (precision 63 %) on the
   blind phrasings under `default`. This is driven by retained exact age, family counts written in compact note
   form, hobbies, work patterns and body intervals. `age_bands` reduces it significantly (−8.3 pp blind) and
   `strict_dates` further (−3.0 pp).
4. **An informed attacker (C) with partial diagnoses and medical chronology re-identifies 34–38 % under
   `default`.** It reaches 18–23 % under the age-band profiles, with high precision. The main evidence is retained
   clinical event years (YEAR = keep), ICD-10 codes (unchanged by task requirement), birth year and age, height and
   weight intervals, and sex. `age_bands` gives a consistent, significant reduction (−14 to −17 pp).
   `strict_dates` adds no significant benefit for C, because the event years that drive it are isolated years,
   which this profile keeps.
5. **Absent targets are rarely falsely attributed** (≤ 4.4 % at threshold 0.9). However, precision is **not
   calibrated across distributions**: B dropped to 59–63 % precision on the blind world. The FS posterior must not
   be read as a probability on new text styles.
6. **Not established.** Risk on real patients or real auxiliary data. The effect of longer, real-style
   encounters (synthetic dialogues have a median of 1.7 k characters against about 6.2 k in the task data). Any
   statement about Bosch's evaluator.

## 4. Clinical cost of the policies (same records; substring checks, conservative)

| Kept exactly (dev, 600 encounters) | default | age_bands | strict_dates |
|---|---|---|---|
| Diagnoses / medications+doses / durations | 100 / 100 / 100 % | same | same |
| Clinical event years ("diagnosed 2015") | 99.9 % | 99.9 % | 99.9 % |
| **Year of full-date clinical events** ("diagnosed 03/12/2014") | 100 % | 100 % | **24.7 %** (45/182) |
| Lab values exact / clinical category kept | 2.4 % / 100 % (by design) | same | same |
| **Blood pressure "128/82"** | **50 %** (92/183) | 50 % | 50 % |
| Weight / height where clinically needed (paediatric, weight-based dosing) | 89 % / 86 % | same | same |
| Paediatric exact age | 100 % | 100 % | 100 % |
| ICD codes, keys, speaker turns, note line count | identical 600/600 | same | same |

The age-band profiles cost no clinical element measured here beyond adult age precision. `strict_dates` deletes
the year of three-quarters of full-date clinical events and does not reduce attack C significantly. On these data
it is not justified. The blood-pressure loss and the needed-measurement generalisation are detector or exemption
issues present under every policy. The blind world shows the same pattern; event years read 94 % there because the
checker does not parse spoken years ("twenty fifteen"), which is a checker limitation.

## 5. Vulnerabilities found on synthetic dev data, classified

| # | Finding (dev evidence) | Class | Affects |
|---|---|---|---|
| V1 | Bare numeric age answer after "how old are you" stays exact under `age_bands` (~25 dev1 encounters) | detector miss | age bands |
| V2 | Spoken DOB with a 2-digit year (`3/4/78`) becomes `[DOB] 78` or `[DATE] 78` under `age_bands` | insufficient transformation | birth year |
| V3 | Adult birth year kept under `age_bands` when no patient age is detected (`DOB: [DOB] 1973`); 24 % of birth-year strings survive under `age_bands` on the blind world | insufficient transformation | birth year |
| V4 | Clinical event years and full-date event years drive attack C | intentional retention (YEAR/DATE keep_year) | C |
| V5 | ICD-10 codes are a strong linkage key for a medically informed attacker | intentional (task requirement) | C |
| V6 | Visit-header year `[DATE] 2025` kept (`default`, `age_bands`); weak evidence here (log2 BF ≈ +0.5) because visits span only 4 years | intentional retention | C |
| V7 | Family counts ≥ 3 in compact note form ("Married, 4 children."), and "our three kids" in dialogue, stay exact | detector miss | B, C |
| V8 | Hobbies, work patterns (night shifts) and marital status are never transformed | intentional (no policy category) | B, C |
| V9 | Residual names (~1–2 %), towns (6 %) and occupation titles (~1 %) | detector miss | all |
| V10 | `BP 114/71` replaced by `[ORGANIZATION]` in ~50 % of vitals lines | false positive (clinical cost) | utility |
| V11 | Clinically needed weight/height generalised in 11–14 % | insufficient exemption | utility |

## 6. Comparison with the teammate's Red Team V2 (read-only, not merged)

RT2 (`D:\download_dada\…\MedAnon-RedTeam-V2`, revision `2dda70a`) was inspected and run unmodified through
`interop/redteam_v2_adapter.py`, on the same anonymised outputs, registers and pools.

| | RT2 (as published) | This benchmark |
|---|---|---|
| MedAnon run | rules-only detectors (no spaCy, BERT or OpenMed) | production detectors, GPU |
| Records | one fixed note template; every quasi-identifier always present; DOB always 15 January; one-line dialogue | Synthea timelines, ASR dialogue + sectioned notes, uncertain dates, probabilistic exposure, dev/test template families |
| Population | invented: 6 jobs, 5 regions, 5 hobby groups, 4 diagnosis groups, an administrative registration year | Synthea: ~350 towns, 70 jobs, 80 conditions, real comorbidity and age structure, doppelgangers |
| Extraction | patterns tied to its own template ("Registered on", "Works as a X.", "Lives in X."); age relative to 2026, not the visit year | generic patterns, birth-year intervals over the visit window, dictionary search over the register |
| Matching | FS-inspired, fixed m = 1 − declared error, frequency-based u, LLR bounded to [−6, 6], names never used | Splink EM (no labels), interval-aware levels, pool posterior, names used if residual |
| Decision | score/margin thresholds chosen on 100 dev encounters | posterior 0.9 fixed a priori |
| Uncertainty | Wilson CIs | patient-cluster bootstrap; paired policy differences |

**Same outputs, same pools (dev, P = 2,000, RT2 default threshold).** On `original`, RT2 reaches top-1 21–27 %
(this attacker: 88–99 %), mainly because it ignores names and finds geography in 2 % of records. On `default` C,
RT2 reaches top-1 4–5 % and success 0–1.3 %, with precision 0–31 % and 4.5 % false matches on absent targets. On
`age_bands` C, its success is 0.7 %. This is under its untuned default threshold; ranking metrics are
threshold-free and show the same gap.

**Findings in common.** Both attackers find that adult age bands help, that a public attacker is defeated, and
that the risk concentrates in informed attackers. **Disagreement.** RT2's large date-policy effect comes from its
artificial, always-present registration year. With realistic dates, administrative years are weak evidence (V6) and
the residual risk sits in clinical event years and ICD codes, which `strict_dates` does not remove. RT2's own
report shows threshold overfitting: at error 0.1, dev precision is 91–92 % and test precision is 72–95 %.
**Worth reusing in a future attacker v2.** Bounded per-comparison LLRs: EM here produced near-veto weights such as
−248 for a matching occupation title, which makes this attacker conservative. Also reusable: error-rate sweeps,
leave-one-group-out attribution, and the explicit dev → test threshold-transfer table. Both codebases stay
separate. The adapter only defines the shared contract: anonymised JSONL + register in, ranked keys + decision out,
scored by one evaluator.

## 7. Future MedAnon hardening plan (to test first; nothing has been changed)

Ordered by expected privacy gain per unit of clinical loss. Each item needs a paired run on a fresh synthetic set,
not on these dev worlds or the blind world.

1. **Close the birth-year leaks under `age_bands`** (V2, V3). Remove the adult DOB year whenever the patient is
   not known to be a child, including 2-digit and spoken years. Expected: the birth-year evidence disappears for
   B and C, at no clinical cost.
2. **Detect bare age answers** (V1): a number in the turn after an age question. Zero clinical cost.
3. **Family counts in note form and in dialogue** (V7): "Married, 4 children", "our three kids". Low cost.
4. **Fix the vital-sign false positive** (V10): `BP` followed by a blood-pressure pattern must never become
   ORGANIZATION. This is a pure clinical gain, and the largest measured utility loss.
5. **Clinical event years: test, do not adopt.** Generalise isolated past-event years to coarse intervals, or to
   relative form against the visit year ("~10 years before visit"), *only* for chronology outside a recent
   window. Expected to cut attack C most. The clinical cost (chronology) must be reviewed with clinicians.
   `strict_dates` as defined is not the right instrument: it costs full-date event years and leaves isolated years.
6. **Clinically needed measurements** (V11): extend exemptions (paediatric vitals and dosing context) and check
   them against clinician-labelled synthetic cases.
7. **Optional, policy decision:** hobby and work-pattern generalisation (V8). Small effect here, against
   occupational-exposure relevance.

## 8. Limitations

- One author wrote the generator and the attacker. Template families and generic extraction patterns reduce, but
  do not remove, shared assumptions. The blind world was evaluated once.
- The synthetic encounters are shorter and cleaner than the task data. Synthea's population is one US state.
- Threat-model availability and error rates are assumptions; the incomplete-register control shows that results
  are sensitive to them.
- The FS posterior is not calibrated across text distributions. EM yields extreme weights for rare levels;
  unobserved levels are neutralised (logged in `attack.json`).
- Splink 5.0.0 drops a comparison after some EM sessions. The model is rebuilt from per-session snapshots, and the
  event is logged (`rebuilt_from_snapshots`).
- Attribute inference for C lacks a knowledge-only baseline. Clinical checks are substring proxies. "Needed"
  measurements follow simple rules (paediatric, weight-based dosing).
- Dev worlds share one population. The bootstrap clusters patients across them but ignores generator-seed
  variance; there are only 2 dev seeds and 1 blind seed.
- No LLM-based attacker was built. The deterministic and FS baselines were the priority, and the knowledge
  ablation suggests that knowledge, not extraction, limits the attack.

## 9. Readiness for a genuinely blind evaluation

The pipeline is ready to be run by someone else on a new synthetic world. The steps are: generate with a new
population seed and a third template family written by another person, run MedAnon at a frozen commit, run the
frozen attacker, then evaluate once. To be cleaner, that evaluation should also (a) add an independently written
template family, (b) add a knowledge-only inference baseline, and (c) apply bounded log-BFs as a pre-registered
attacker v2.
