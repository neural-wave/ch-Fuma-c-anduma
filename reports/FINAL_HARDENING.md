# MedAnon — final production hardening (18 Oct 2026)

Local branch `hardening` (from `reid-benchmark` `f5503db`, which contains production `ceff209`). Nothing was
pushed, merged or submitted. **All numbers come from synthetic data or local aggregate proxies. None of them is an
official Bosch score.** The previously reported Bosch figures (privacy removal 0.5918, ICD-10 F1 delta 0.0, concept
recall 1.0) can only be re-measured by an authorised Bosch evaluation.

## 1. Versions compared (identical conditions)

| | Baseline | Candidate |
|---|---|---|
| Code | `ceff209`, code sha256 `ac834ee0…` | `hardening`, final src `b698ca8`, code sha256 `400b20f0…` (identical locally and on the GPU host). Paired GPU outputs come from `f72e1ea` (`bd502bf4…`). The only later src change narrows the landmark rule to names without clinical association; it produces the same matches on every synthetic input (checked rule by rule) and the GPU suite was re-run on `b698ca8`. |
| Config | `configs/age_bands.yaml`, sha256 `dedfed9e…` | same file, same hash |
| Stack | rules + spaCy lg/Presidio + dslim BERT (cascade) + OpenMed PII 434M, `--offline --device cuda`, `MEDANON_SECRET` fixed | same |
| Hardware | Vast RTX PRO 5000 (48 GB), `/venv/main`, cached weights `/workspace/hf` | same, same session |

Inputs are fictional only: the Synthea worlds `dev1` and `dev2` (300 encounters each), `long1` (60 long-form
records), the synthetic rounds 2/3/4 DEV20 and 6 DEV48 with their gold, and the teammate's 12 probes.
`long1` reaches task-like length: median dialogue 6,420 characters against 6,196 in the task dev split, median note
2,583 against 2,648 (aggregates only).
The dev1/dev2 baseline outputs were reused from the re-identification benchmark (byte-identical, sha256
`08b12ba2…`). The Round 5 holdout, Round 6 validation gold and `test1` were **not** opened or used.

## 2. Vulnerabilities investigated

| # | Defect (fictional repro) | Root cause | Fix (minimal, in the existing architecture) | Outcome |
|---|---|---|---|---|
| A | `BP 114/71` → `[ORGANIZATION]` | spaCy lg tags "BP …" at line start as ORG (a company); `propagation` copies the span to every "BP …" | `_clean_statistical`: reject a statistical ORG/LOC/PERSON/NATIONALITY span made of a vital-sign abbreviation followed by a value, or of a BP pattern | **fixed** |
| A' | `SOCIAL HX` → `[ORGANIZATION]` (found while auditing A) | same spaCy ORG on section headers | reject spans that form a clinical section-header line | **fixed** |
| V2 | `lungs`/`neurovascular`/`biofreeze`/`referral` → identifiers; "kidney doctor" → OCCUPATION (teammate finding) | NER on clinical words; missing lexicon entries | reject statistical spans made only of clinical words with no name evidence, and care-team roles labelled OCCUPATION; 7 real clinical terms added to `clinical_terms.txt` | **fixed** (probes SYN-OVER-01/03 pass) |
| B1 | Adult birth year kept under `age_bands` when no age is detected (`DOB: [DOB] 1973`) | removal depended on a detected patient age | `Transformer`: under the band policy, the DOB year is removed when reference year − birth year ≥ 18, decided from the year itself. A child's year is kept, and an undecidable year is removed. | **fixed** |
| B2 | `3/4/78` → `[DOB] 78` / `[DATE] 78` | 2-digit year; a filler ("okay ,") broke `DOB_ANSWER_RE`, so the date fell back to DATE and kept its year | `birth_year()` parses 2-digit and apostrophe years; DOB answers accept common fillers; `born in '68` | **fixed** |
| C | `how old are you now ?` → `[patient] 49` stays exact | `AGE_ANSWER_RE` accepted only "how old are you ?" with nothing in between | generic age question (words before "?", "you're how old", "what's your age"), fillers in the answer, relatives excluded | **fixed** |
| D | `Married, 4 children.` / `our three kids` stay exact | missing cues in `FAMILY_RE` | marital-status and possessive cues; hereditary exemption extended ("have/has … disease") | **fixed** |
| E1 | Towns after a residence cue (`Lives in Cambridge.`) | the "all-common-words" ORG/LOC filter dropped spaCy LOCATION spans | keep LOCATION after a residence cue (lives in, moved to, drive in from …) | **improved** (see §3) |
| E2 | Partial names (`you must be [PERSON] waters ?`) | the remaining surname is also a common word | none: lexical signals (name ratio, frequency) cannot separate *waters/marks/hills* from *will/may/said*, so a rule would over-redact | **documented** |
| V4 | `mount hood` next to a masked state (teammate finding) | no landmark rule | Mount/Mt/Lake + a rare word with no clinical association, `<Name> National Park/Forest`; clinical criteria excluded | **fixed**. On the real eval split the first version fired once on an ambiguous "mount + clinical-lexicon word" and removed one concept; the rule was narrowed and now fires 0 times there |
| A1 | `code_sha256` differs per machine (teammate finding) | `code_hash()` also hashed a nested virtualenv | skip `env/`, `venv/`, `site-packages/`, `__pycache__/` | **fixed**: local hash = GPU hash `bd502bf4…` |

## 3. Paired before/after (same inputs, same scorers, same denominators)

**Targeted leaks, Synthea worlds (baseline → candidate).** Denominators: dev1 300, dev2 300, long1 60 records.

| Measure | dev1 | dev2 | long1 |
|---|---|---|---|
| Blood pressure kept exactly (`clinical:vital`) | 44/92 → **92/92** | 48/91 → **91/91** | 18/47 → **47/47** |
| Section headers kept | 2793/2806 → **2806/2806** | 2717/2737 → **2737/2737** | 1145/1163 → **1163/1163** |
| Adult encounters with a bare exact age answer | 31 → **0**¹ | 29 → **0** | 28 → **0**¹ |
| Adult encounters with a 2-digit DOB (`[DOB] 78`) | 6 → **0** | 4 → **0** | 4 → **0** |
| Adult birth year visible | 7 → 1² | 8 → 3² | 3 → 2² |
| Exact household count ≥ 3 visible (patients with ≥ 3 children) | 20/61 → **0/61** | 31/54 → **0/54** | 12/16 → **0/16** |
| Residual town (of encounters naming it) | 5/104 → 3/104 | 8/103 → 4/103 | 2/42 → 1/42 |
| Residual first / last name | 4/162, 3/261 (unchanged) | 2/156, 3/251 (unchanged) | 2/52, 0/60 (unchanged) |
| Full birth date / exact adult age `N-year-old` / occupation title | 0/116 · 0/206 · 2/161 (unchanged) | 0/127 · 0/199 · 2/152 (unchanged) | 0/64 · 0/10 · 0/55 (unchanged) |
| Diagnoses / medications+doses / durations | 100 % / 100 % / 100 % (unchanged) | unchanged | unchanged |
| Event years / full-date event years | 716/718, 93/93 (unchanged) | 650/650, 89/89 (unchanged) | 599/608, 67/68 (unchanged)³ |
| Lab category kept / exact lab values | 305/305, 6/305 (unchanged, intervals by design) | unchanged | unchanged |
| Weight / height needed exactly (paediatric, weight-based dosing) | 67/74, 34/39 (unchanged) | 62/71, 33/39 (unchanged) | 12/12, 6/6 (unchanged) |
| Record order, ids, keys, `icd10_codes`, speaker turns | identical (unchanged) | identical | identical |

¹ The 2 remaining matches are `90+`, which is correctly top-coded.
² All remaining cases are a clinical event diagnosed in the birth year ("Atopic dermatitis, dx 1998"). This is
clinical chronology kept by policy; see §7.
³ The misses are spoken years that the checker cannot parse, the same in both versions.

**Synthetic rounds against gold** (`scripts/score_gold.py`; R2/R3/R4: 20 records each; R6: 48)

| | R2 | R3 | R4 | R6 |
|---|---|---|---|---|
| PHI recall (raw), baseline = candidate | 261/269 | 257/281 | 234/239 | 1178/1190 |
| Policy-compliant recall | identical | identical | identical | identical (1190/1190) |
| Precision (core denominator) | 269/272 = | 287/288 = | 256/258 = | 1268/1646 → 1269/1647 |
| Clinical traps preserved | identical | identical | identical | identical |
| Encounters with a direct identifier left | 0/20 = | 7/20 = (gold ID-cue convention, pre-existing) | 0/20 = | 0/48 = |
| Output text changed | 0/40 fields | 0/40 | 0/40 | 1/96⁴ |

⁴ In the baseline, an over-wide facility span also swallowed "on <date>". The candidate publishes `[DATE] 2026`,
as the DATE policy specifies. This is the only change, and it explains the R6 strict-token DATE proxy moving
0.021 → 0.0.

**Independent checks**

| Check | Baseline | Candidate |
|---|---|---|
| Teammate probes (12 encounters): privacy / clinical | 7/12 · 39/42 | **12/12 · 42/42** |
| Teammate `audit_masks`: bare age answers (dev1/dev2/long1) | 18 / 15 / 16 | **0 / 0 / 0** |
| `audit_masks`: birth years visible, 4-digit + 2-digit (dev1/dev2/long1) | 31 / 32 / 3 | 23 / 22 / 0, **all children** (policy) |
| `audit_masks`: clinical-term over-redaction blocks | 0 | 0 |
| Frozen Splink attacker (unchanged, hashes verified), threat C, P = 2,000, dev pooled: success | 21.5 % (17.8–25.3) | **11.7 % (9.0–14.5)**; paired −9.8 pp (7.2–12.6); open world 15.0 → 5.6 % |
| Frozen attacker, threat B: top-1 / deterministic success | 10.2 % / 4.2 % | 4.7 % / 2.5 % (paired −1.7 pp, 0.7–2.8) |
| Frozen attacker, false matches on absent targets (A/B/C) | ≤ 0.4 % | ≤ 0.8 % |

The attacker figures are dev worlds already used to develop that attacker: a secondary diagnostic, not a blind result.

**Real eval split (67 Bosch-provided records, run locally on CPU, never uploaded; aggregates only)**

| | Baseline | Candidate |
|---|---|---|
| `medanon validate` | ok, 67/67 | ok, 67/67, no problems |
| ICD-10 proxy F1 (TF-IDF trained on dev originals): original → anonymised | 0.3836 → 0.3836, delta 0.0 | 0.3836 → 0.3836, **delta 0.0** |
| ICD description terms retained | 246/246 (1.0) | 246/246 (1.0) |
| Concept recall, lexical proxy (set / mention) | 0.9945 / 0.9938 | 0.9942 / 0.9939⁵ |
| Name-seed leaks | 0/67 | 0/67 |
| Teammate `audit_masks`: clinical-term over-redaction blocks (records) | 7 (6) | **3 (3)** |
| `audit_masks`: exact adult age / bare age answer / adult birth year | 0 / 0 / 1 | 0 / 0 / **0** |

⁵ Mention level: the candidate keeps 7 concept occurrences the baseline lost, and loses 2. One of the 2 came from
the first landmark rule, now narrowed (0 matches on this split). The other has no overlapping candidate decision.
These local proxies are not Bosch's concept-recall metric.

## 4. Tests

- Full suite on the GPU host with the production detectors, final src `b698ca8`: **191 passed, 0 failed**
  (176 historical + 15 new).
- New `tests/test_hardening.py` has 15 tests, with positive cases and clinical negatives:
  - **positive cases:** vital forms, headers, clinical words, adult/2-digit/filler DOB, bare ages, compact
    family size, residence towns, landmarks, code hash;
  - **clinical negatives:** a real employer is still masked; the paediatric birth year and the clinical years and
    dates are kept; pain score, home BP, duration and a relative's age are kept; counts 0–2 and hereditary counts
    are kept; "mount the brace" and "lake level" are kept;
  - **teammate probes:** the 12 fictional encounters, copied to `tests/fixtures/redteam_handoff/`.
- On baseline `ceff209` the same new tests fail 8/11 (spaCy profile), which confirms that the fixes are what makes
  them pass.

## 5. Performance (GPU, production stack)

| | Baseline | Candidate |
|---|---|---|
| Device VRAM peak (nvidia-smi, 200 ms, idle device, includes CUDA context) | 2,639 MiB | **2,639 MiB** (every dataset ≤ 2,639) |
| Process RSS peak | 3.64 GB | 3.64 GB |
| Latency incl. model load: dev worlds / long1 | 0.26 / 0.81 s per record | 0.26 / 0.79 s per record |

The peak leaves about 1.4 GB of margin under 4 GB. A physical 4 GB card was **not** tested: the CUDA context,
allocator and driver may differ on one.

## 6. Clinical regressions

- **None found** in any measured category: diagnoses, medications and doses, durations, clinical years and dates,
  lab categories, needed measurements, ICD payload, gold clinical traps R2–R6, probe clinical assertions, and
  ICD proxy F1 on the real eval.
- On the real eval split, concept recall moved by ±0.0003 at most (note 5). The mention level is net positive.
- **Restored:** blood pressure, which went from about 50 % kept (92/183 on dev1 + dev2) to 100 %; section headers;
  clinical words wrongly
  masked as identifiers.
- **Unresolved, pre-existing and unchanged:**
  - About 9–14 % of clinically needed weights and heights are generalised. The exemptions follow rules (paediatric,
    weight-based dosing), not clinical understanding.
  - Exact lab values become category intervals by design. Note that `EF [40-50]` straddles the HFrEF/HFmrEF
    cut-off.

## 7. Remaining privacy limitations (intentional or unresolved)

- **Intentional retention:**
  - Clinical event years, including a diagnosis made in the birth year, which reveals that year.
  - ICD-10 codes, as the task requires.
  - The visit year (`[DATE] 2025`).
  - Exact paediatric ages and children's birth years.
  - Hobbies, work patterns and marital status.
  - Adult age bands, which reveal a decade.
- **Unresolved detector misses:**
  - Partial names: about 1–2 % of name mentions on synthetic data, typically a surname that is also a common word.
  - Some towns without a residence cue, or dropped by other filters (for example "North Reading").
  - Occupation titles: ~1 %.
- **Residual linkage.** On synthetic data, an attacker who already knows partial diagnoses and chronology (threat C)
  still re-identifies ~12 % of dev encounters at P = 2,000. ICD codes and event years drive this, and the policy
  keeps them.

## 8. Reproduce the final output

```bash
git checkout hardening        # final local commit, see §10
pip install -r requirements.txt && medanon prepare-models   # once, with network
medanon anonymize --offline --device cuda --config configs/age_bands.yaml \
    -i data/eval_patient_doctor_encounters.jsonl -o outputs/eval_patient_doctor_encounters.anonymized.jsonl
medanon validate --original data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/eval_patient_doctor_encounters.anonymized.jsonl
```

Use `--device cpu` without a GPU (about 15 s per record). Set `MEDANON_SECRET` only to reproduce lab perturbation;
the profile uses intervals, so it does not matter for this profile. The paired experiment scripts are
`research/reid/scripts_hardening_eval.py` and `research/reid/scripts_hardening_attack.py`. The GPU scripts and logs
are in `D:\NEURAL_WAVE\_cache\hard_runs` (local copy) and `/workspace/hard` (instance).

## 9. Is the submission profile applied automatically?

**No.** Without `--config`, the built-in default still publishes exact adult ages, keeps adult birth years and
keeps compact household counts below the band policy. The README now shows the submission command with
`--config configs/age_bands.yaml`. The global default was **not** changed. Changing it would make the repository
safe to run without flags, but it needs your approval: it changes documented behaviour, and the policy tests in
`tests/test_policy.py` assert the exact-age default.

## 10. Actions requiring human approval

1. Freeze this candidate (branch `hardening`, final commit in `git log`).
2. Decide whether to make `age_bands` the built-in default, or keep the explicit `--config`.
3. Run the one-time post-freeze validation (Round 5 holdout / Round 6 validation gold).
4. Generate the final eval JSONL from the frozen commit with the command in §8. A local CPU run already exists
   for checking only (`D:\NEURAL_WAVE\_cache\hard_runs\real\`); it is not a deliverable.
5. Any push, merge, publication or Bosch submission.

## Independent Red Team Handoff Review

- **Package.** `D:\download_dada\medanon_attacker_handoff\medanon_attacker_handoff` (extracted folder, no zip
  present). Attacker repo `progetto-hackaton@adversary-reidentification`, commit `102040f8…`, clean tree; tested
  MedAnon `ceff209`. Inspected read-only.
  - Its deterministic `audit_masks` ran in a separate venv (`pydantic`, `rapidfuzz`, `requests`), in `-I` mode,
    with no network.
  - Its LLM attack (35B, Ollama) was not run: it is not needed to act on the findings.
- **Already known:**
  - V3 (exact adult age under the default): `age_bands` addresses it.
  - V1 (adult birth year) = my B1/B2.
  - V5 (bare age answer) = my C.
- **New and verified:**
  - V2 (clinical words masked as identifiers; 13/140 real records by their aggregate audit): reproduced by
    probes SYN-OVER-01/03 at `ceff209`, fixed.
  - V4 (landmark next to a masked place): reproduced by SYN-LANDMARK-01, fixed.
  - A1 (machine-dependent code hash): confirmed, fixed.
- **Their "BP → [ORGANIZATION] not reproduced".** Consistent with my analysis: the defect needs a vitals-line
  format ("BP 122/70, HR 81" at line start). None of their 9 forms used it, and their 140 real records showed 0
  cases. Real prevalence may therefore be low, but the defect is real (91/183 synthetic vitals lines lost in
  dev1+dev2) and it is fixed.
- **Incorporated.** Their 12 probes and 54 assertions are a permanent regression test. Their gate is met: clinical
  39/42 → 42/42 with no previously-passing assertion lost, privacy 7/12 → 12/12, exact adult ages 0, adult birth
  years 0, clinical-term blocks not higher.
- **Unresolved from their list.** "barky cough → [EMPLOYER]" and "lungs → [PERSON]" were seen only in real
  transcripts, and their own probes did not reproduce them (SYN-OVER-02/04 pass). The V2 filter targets this
  mechanism, but real-data confirmation needs their aggregate audit re-run on the 140 training records, which is
  local-only.
