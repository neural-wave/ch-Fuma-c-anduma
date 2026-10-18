# Handoff to Claude Code: re-identification attacker → MedAnon fixes

> Privacy note (2026-10-18): real encounter IDs are replaced by neutral labels (R1–R10, T1–T3), and record-specific names, places and verbatim quotes by paraphrases. Per-record files keyed by real IDs stay local.

To the agent improving MedAnon: this package gives you an independent attacker, measured findings
against **MedAnon `ceff209`**, synthetic regression probes and a non-regression gate. You should not need
to rebuild or reverse-engineer anything. Numbers come from `results/*.json` and
`synthetic_regressions/outputs/*.check.json`, and `VULNERABILITIES.json` is generated from them.

## 1. What this attacker is (and is not)

| component | what it does | needs |
|---|---|---|
| `attack.run` | LLM reads each **anonymized** record, extracts the patient's quasi-identifiers with verbatim evidence, prices each against US population tables (Census/BLS) into bits, λ = expected look-alikes, risk = (1−e^−λ)/λ; a second prompt guesses what each placeholder hid | Ollama + `qwen3.5:35b` (~22 GB VRAM) |
| `attack.audit_masks` | **deterministic, no LLM**: aligns original↔anonymized, classifies what every placeholder replaced (identifier, name, date, clinical term, measurement), scans anonymized text for exact adult ages, bare age answers, 4/2-digit birth years | originals (dev/synthetic only) |
| `synthetic_regressions/` | 12 invented encounters, 54 assertions (12 privacy, 42 clinical), actual `ceff209` outputs for 3 policies | MedAnon |
| `attack.export_predictions` | attacker-side predictions only, for comparison with your Splink attacker | a `per_record.jsonl` |

**Not computed, so don't look for them:** Top-1/Top-5, linkage success, coverage, precision, wrong
attributions, absent-target false matches. There is no synthetic population and no auxiliary identity
database: the auxiliary knowledge is population frequency tables plus the LLM's world knowledge. All
experiments used the **full production detectors**; no rules-only runs.

## 2. What was actually measured (provenance matters)

| evidence | MedAnon build / policy | n |
|---|---|---|
| LLM attack, eval R1–R5 | **older build** (code `12cb9984…`, config `d85e6375…` matches nothing at `ceff209`), exact ages | 5 |
| LLM attack, train T1–T3 | `ceff209`, built-in defaults (no `--config`) | 3 |
| `audit_masks`, 140 training encounters | `ceff209` × {defaults, `age_bands.yaml`, `age_bands_strict_dates.yaml`} | 140 × 3 |
| synthetic probes | `ceff209` × same 3 policies | 12 × 3 |

`ceff209` = `src/medanon` and `configs/` byte-identical on branch `adversary-reidentification`.
Its local `code_sha256` is `4ff71990…`, not your `ac834ee0`: `manifest.code_hash()` also hashes the
git-ignored virtualenv `src/medanon/env/`, so the same commit hashes differently per machine (finding A1).
The LLM results are on 8 Bosch-provided records only: indicative, not statistics. The deterministic
140-record audit is the strongest evidence here.

## 3. Policy comparison (140 training encounters, `ceff209`, same input)

| metric (`attack.audit_masks`) | defaults | age_bands | age_bands_strict_dates |
|---|---|---|---|
| exact adult age visible | 109/140 | **0/140** | 0/140 |
| mean age information, max per record (bits) | 5.25 | **2.75** | 2.75 |
| birth year visible (4-digit / 2-digit) | 20 / 2 | 3 / 1, all children (kept by design) | 3 / 1, all children |
| full date with visible year (`[DATE] 2021`) | 21/140 | 21/140 | **0/140** |
| records with clinical-term over-redaction | 13/140 | 13/140 | 13/140 |
| synthetic probes privacy / clinical pass | 4/12 · 39/42 | 7/12 · 39/42 | 7/12 · 39/42 |
| processing time (4 workers, Mac CPU) | 2.54 s/record | 2.46 s/record | 2.32 s/record |

- **Observed privacy benefit:** `age_bands` removes every exact adult age and roughly halves the age
  information (one-year cohort ≈ 6.3 bits → decade band ≈ 3 bits). It is the largest measured gain.
- **Clinical cost:** decade bands lose age-specific thresholds (screening at 45 vs 50, dosing).
  Paediatric ages stay exact.
- **`strict_dates`:** its only measured difference from `age_bands` is removing the year from full dates
  (21/140 → 0/140 records). My attacker scores a year-only date as 0 bits, so its privacy value depends on
  linkage against dated records, which your Splink benchmark can measure and mine cannot. The clinical cost
  is the calendar year of dated events in 21/140 records. Too uncertain to recommend on my evidence alone.

## 4. Findings, ranked (full detail and probes in `VULNERABILITIES.json`)

| id | finding | type | evidence at `ceff209` | severity |
|---|---|---|---|---|
| **V3** | exact adult age under default policy | intended policy | 109/140 records; largest single quasi-identifier in all 8 LLM-attacked records (≈6.3 bits) | high → fixed by `age_bands` |
| **V2** | clinical words masked as identifiers: `lungs`→[PERSON] ×5 records, `neurovascular`→[LOCATION] ×3, `barky cough`→[EMPLOYER], `Biofreeze`/`referral`→[ORGANIZATION], specialties→[OCCUPATION] ×2 | incorrect transformation | 13/140 records, every policy; synthetic SYN-OVER-01/03 reproduce | medium (clinical integrity) |
| **V1** | adult birth year kept under `age_bands` when no adult age is detected in the record (`[DOB] 1971`, `[DOB] 71`); removed when an age is stated | incorrect transformation (contradicts the config's own comment) | synthetic SYN-DOB-01 fails / SYN-DOB-02 passes; real 0/140 (age always stated in these transcripts) | high per instance, rare |
| **V4** | named landmark left next to a masked place (`[US_STATE] … mount hood`) | detection miss | LLM recovered the masked state from a named landmark (eval, older build); synthetic reproduces; 0/140 train | medium per instance, rare |
| **V5** | bare answer to "how old are you?" kept (`[patient] 54 .`) | detection miss | synthetic reproduces under all policies; 0/140 train | high per instance, rare |
| A1 | manifest `code_sha256` machine-dependent | evaluation artifact | explained above | low |

Leads from your attacker I checked:
- **"BP → [ORGANIZATION]": not reproduced.** 0 of 9 blood-pressure forms (`142 over 88`, `128/82`, `bp is
  150 over 95`, `one forty over ninety`, …) and 0 measurement masks in 140 records.
- **"Over-redaction of values/dates": mostly intended policy.** Lab values are generalized (`EF 45%` →
  `[40-50]%`, `A1c 8.2` → `[8-9]`). Medications, doses and relative chronology survive. My attacker never
  prices labs, so it has no evidence on lab linkage either way. Note that `[40-50]` straddles the HFrEF
  (≤40) / HFmrEF boundary.

## 5. Hardening plan (max five, by expected value)

| # | change | evidence | privacy gain | clinical impact | effort | when |
|---|---|---|---|---|---|---|
| 1 | Ship `configs/age_bands.yaml` | V3: exact ages 109→0/140, age bits 5.25→2.75 | largest measured | decade bands | config only | **before submission** |
| 2 | Clinical-vocabulary guard on detector spans (anatomy/exam words, symptoms, OTC products, "referral", specialty nouns never become PERSON/LOCATION/ORG/EMPLOYER/OCCUPATION) | V2: 13/140, all policies | none needed | **positive**: restores exam findings and symptoms | small (lexicon + filter; MedAnon already has medical lexicons) | **before submission** |
| 3 | Decide "adult birth year" from the year itself (reference year − year ≥ 18 → drop year), parse 2-digit years | V1 synthetic | closes the age_bands bypass | none for adults | small | before submission if 1–2 are done |
| 4 | Mask named natural landmarks (Mount/Mt X, Lake X, X National Park, X River), at least in sentences with a masked place | V4 synthetic + 1 real LLM recovery | removes a demonstrated recovery path | none expected; negative tests for eponyms/scores | small–medium | if time allows, else document |
| 5 | `code_hash()` must skip `src/medanon/env/` (or hash git-tracked files only) | A1 | none (provenance) | none | trivial | before submission |

Optional sixth, cheap but rare: a dialogue-turn rule for bare age answers (V5).

## 6. Non-regression gate for any MedAnon change

1. `python synthetic_regressions/check_regressions.py <anon of encounters.jsonl>`: every clinical
   assertion that passes at `ceff209` (39/42) must still pass, and the privacy pass count must not drop.
   A fix is confirmed when its probe's privacy assertions turn to pass (V1: SYN-DOB-01; V2: SYN-OVER-01/03
   clinical; V4: SYN-LANDMARK-01; V5: SYN-AGE-02).
2. `python -m attack.audit_masks --original <dev orig> --anonymized <dev anon> --out a.json`: exact adult
   ages stay 0, adult birth years 0, and `clinical_term` blocks must not increase (target 0).
3. Your own Bosch-style checks unchanged: ICD-10 F1 delta 0.0, medical-concept recall 1.0.
4. No tuning on blind validation records. Synthetic and dev only.

## 7. Comparing with your Splink attacker

- **Input contract:** MedAnon output JSONL, one object per line: `encounter_id`, `dataset`, `dialogue`,
  `note`, `icd10_codes` (list of `{code, description}`). Placeholders match
  `\[[A-Z][A-Z_]*(?:_\d+)?(?::[^\]]*)?\]`.
- **Run both attackers unchanged on the same file.** Mine:
  `python -m attack.run --input anon.jsonl --out results/x` (add `--original orig.jsonl` only for scoring).
- **Truth isolation:** the extraction and recovery prompts receive only the anonymized text. The original
  is read only by the scorer (alignment ground truth, recovery matching, and the comparison union; facts
  borrowed from it are flagged `from_original`). `attack.export_predictions` strips all of that and
  re-scores risk from the attacker's own facts:
  `{encounter_id, risk, lambda, bits, quasi_identifiers:[{category,value,evidence,p,p_source,bits,counted}], recovery_guesses:[{placeholder,type,guess,confidence}]}`.
- **What to compare:** per encounter, the quasi-identifiers each attacker relies on, and whether its
  high-risk encounters coincide with your high Top-k matches. My `quasi_identifiers` can also feed your
  linker as an attacker-extracted profile. Don't merge the codebases.

## 8. Limitations

- The LLM results cover 8 records, and 5 of them come from an older MedAnon build.
- Risk is expected uniqueness under an independence assumption, not real re-identification.
- The rarity buckets for lifestyle facts are heuristic: about 2 bits mean error per record against a
  human review of 8 records.
- Synthetic probes show mechanisms, not prevalence. Prevalence comes from the 140-record audit.
- The regex scans can miss unusual phrasing. They were hand-checked on the flagged items: one false
  positive (a relative's age) was removed.
- Nothing here predicts Bosch's official privacy-removal score (0.5918 reported): different metric,
  different data.

## 9. Package layout

```
FOR_CLAUDE_CODE.txt   CLAUDE_HANDOFF.md   VULNERABILITIES.json   MANIFEST.json (sha256, versions, settings)
attack/               attacker + audit + export, tests (92), README.md (usage, metrics, troubleshooting)
synthetic_regressions/ encounters.jsonl, expectations.json, check_regressions.py, outputs/ (ceff209 × 3 policies)
results/              provenance.json, policy_audit_train140.json, llm_attack_runs.json (aggregates only)
```

Excluded on purpose: Bosch records and anything quoting them (per-record LLM outputs, `REPORT.md`),
caches, model weights.
