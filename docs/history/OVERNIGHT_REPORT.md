# Overnight report — review fixes, model experiments, submission state

_Branch `dev` in `D:\NEURAL_WAVE\Anonymization_Bosch_NeuralWave` (local commits only, nothing pushed or
sent). Checkpoint before the work: `c4fac8c`. Numbers are local proxies, not Bosch scores._

## 1. Independent review — status of every finding

| ID | Finding | Status | Fix (regression test) |
|---|---|---|---|
| P0-01 | Default could silently fall back to another pipeline; no record of what ran | **fixed** | ML detector is `required: true`: a missing model stops the run with instructions instead of degrading. Weights are resolved from the local HF cache; a one-time download is allowed at setup and forbidden with `--offline`. Every run writes `<output>.manifest.json` with code hash, config hash, model revision, weights SHA-256, effective detector status, library versions, input/output SHA-256 and record counts. (`test_required_model_fails_loudly`, `test_cli_writes_manifest_and_validates`) |
| P0-02 | Spoken-digit MRN / phone left in clear | **fixed** | Cue-based rule (MRN, record/account/member number, phone, call me at …) plus a rule for ≥ 7 digit words in a row without any cue; spoken digits also pass the ML ID filter. (`test_spoken_digit_identifiers_removed` ×5; `test_spoken_clinical_numbers_untouched` ×4) |
| P0-03 | Audit wrote original text for kept candidates | **fixed** | No surface in any decision unless `--audit-include-text`. Words of the surface are scrubbed from explanations (`<span>`), and kept decisions carry a `reason_code`. (`test_audit_contains_no_original_surfaces_by_default`, `test_audit_includes_surfaces_only_on_request`) |
| P0-04 | Deliverable / submission readiness | **partly (needs humans)** | Strict `medanon validate`: count, order, unique ids, key order, non-text fields, speaker tags, line structure. Deliverable regenerated with a manifest. Team names, video, 3-page PDF and publishing remain open, see §5. |
| P1-01 | Real privacy may be below the headline proxies | **mitigated / open** | Held-out adversarial set with Wilson CIs, blind Label Studio annotation workflow (`scripts/annotation.py`), proxies labelled as such. Human annotation not done yet. |
| P1-02 | Lab perturbation exceeded 3 % (TSH 0.1→0.2) | **fixed** | Candidates are capped at the configured relative change; values that cannot move are kept and the reason logged. Values inside reported ranges ("EF 55-60%") are never perturbed. (`test_perturbation_never_exceeds_relative_bound`, `test_small_values_are_kept_not_distorted`, range case in `test_clinical_boundaries`) |
| P1-02b | Perturbation keyed on public seed + encounter id (invertible) | **fixed** | Secret key from `MEDANON_SECRET`, otherwise random per run and never stored. Inversion attack on eval (28 perturbed values): exact recovery 42.9 % with the public seed vs 21.4 % with the secret key. Those remaining matches are coincidences: a 3 % bound allows only a few shifts per value. (`test_inversion_attack_defeated_by_secret_key`, `test_perturbation_key_is_not_public`) |
| P1-03 | Lab-linkage attack used exact equality only | **fixed** | Tolerance-based matching (exact / 3 % / 5 %) reported as correct-unique, wrong-unique, target-in-set, no-match. (`test_lab_linkage_with_tolerance_beats_exact_matching`) |
| P1-04 | Linkage metric logic errors | **fixed** | Candidate sets of record indices; zero candidates is no longer counted as success; wrong-unique is separated; the early `break` is removed. (`test_score_candidates_separates_outcomes`) |
| P1-05 | "k-anonymity" claimed without guarantee | **fixed (claim) / limited (method)** | Added a last-resort suppression level and a post-hoc check of class sizes over the declared quasi-identifiers (`qi_verification`). Docs now say "heuristic generalisation of declared QIs, not formal k-anonymity"; free text, dates, ICD codes and labs are out of scope. |
| P1-06 | Utility proxies partly circular | **mitigated** | Independent checks: negation-cue counts, every number outside replaced spans, typed clinical traps (negation, family history, dosage, labs, eponyms, temporal, diagnosis). The lexicon concept recall is still reported but labelled as circular. |
| P1-07 | Shared surnames merged across people | **fixed** | A partial name contained in several full names is not attached to any of them (neutral placeholder); only one entity can be `[PATIENT_NAME]`. (`test_shared_surname_not_merged`, `test_only_one_patient_placeholder`) |
| P1-08 | Overlap union could erase clinical words | **fixed** | A weaker span with a different label cannot extend the replacement over strong medical vocabulary. On dev, medical words inside replacements went from 7 kinds to one: surnames after a title that are also clinical words. Those are names, so replacing them is correct. (`test_overlap_resolution_keeps_clinical_words`) |
| P2-01 | Long turns in the BERT cascade | **fixed** | Regions longer than the window are split into overlapping windows. (`test_long_turn_is_windowed`) |
| P2-02 | Non-US / Unicode names | **fixed (Latin scripts)** | Letter classes cover Latin-1, Latin Extended A/B and Latin Extended Additional (diacritics, Vietnamese). Non-Latin scripts are still out of scope. |
| P2-03 | Policy ablations | open | Documented; relative dates, seasons and ethnicity are policy switches. |
| P2-04 | Efficiency / integrity measurement | **fixed** | Peak RSS sampled over the process *and* worker processes; `gpu_used` derived from the device; strict deliverable validation. |
| P2-05 | Docs / traceability | **fixed** | Manifest per run; README/REPORT updated; `CODEBASE_EXPLAINED.md` flagged as predating the fixes. |

Additional bugs found during this work:
* **Line breaks could be lost.** A date match spanning a newline merged two dialogue lines. Spans are now split per line.
* **Lab range and time-unit checks looked at the wrong position.** They used the end of the whole match instead of the end of the number.
* **The new scorer misparsed "a1c" as the value "1".** Fixed in the scorer.
* **Real surnames had reached the docs twice.** Each time a diagnostic printed a surname as a
  "medical word":
  * in a draft of this report, removed before any commit;
  * in session 1, in the README's concept-recall row. That one was committed and pushed (§5.7).

  New gate `scripts/privacy_check_docs.py`: it checks docs, code and commit messages against the
  name seeds of the local data, and prints masked context only. With `--ref` it checks any branch
  or commit.
  * The `dev` tip: FAIL 0. Its 2 warnings are a very common surname used as a generic example in a
    code comment and in a synthetic test.
  * Illustrative names in the docs that collide with real names were replaced by names absent from
    the data.
* **A relative's age at onset was banded.** "Mother … at age 45" in family history lost its age; it is now kept.

**Tests:** 99 passed (was 51). New files: `tests/test_review_fixes.py` and `tests/test_clinical_boundaries.py`.

## 2. Model experiments (details in `MODEL_EXPERIMENTS.md`)

Same held-out adversarial set (40 encounters, 720 gold identifiers) and same 25 real dev encounters
for every configuration:

* **A, Healthcare-Brain-NER:** recall 0.786 → 0.794, not significant. Not integrated.
* **B, Qwen3.5-2B** (4-bit GGUF, structured JSON, verbatim span validation): 99.4 % valid JSON and
  1.7 % hallucinated spans (rejected). Recall +1.1 pt, not significant, at 57 s per real encounter
  on CPU. Not integrated.
* **C, OpenMed PII SuperClinical:** recall **0.786 → 0.886** (95 % CIs do not overlap) with no
  precision or trap loss, mostly from occupations without a cue. It changed only 2 spans in 25 real
  encounters, at ~12× CPU time. Offered as the opt-in `configs/max_privacy.yaml`.

## 3. Final configuration and deliverables

**Configuration.** The default profile (`configs/default.yaml`): rules + spaCy `en_core_web_lg`
3.8.0 + Presidio patterns + `dslim/bert-base-NER-uncased` cascade on the dialogue, with the model
required. Values from the deliverable's manifest:

| Item | Value |
|---|---|
| ML detector | **active**, revision `301540b48433…`, weights SHA-256 `cb4581a5ae49…` |
| Code / config SHA-256 | `03813eefc5c4…` / `9eae0f9a3f08…` |
| Libraries | Python 3.12.6, spaCy 3.8.16, presidio-analyzer 2.2.364, torch 2.14.1+cpu, transformers 5.16.1 |
| Lab key | random per run, not stored |
| GPU | not used |

**Deliverable** (`outputs/submission/`, git-ignored):

* `eval_patient_doctor_encounters.anonymized.jsonl`: 67 records, SHA-256 `f173dcb66405bac6…`.
* `…anonymized.manifest.json` and `SHA256SUMS.txt`.
* `medanon validate`: **ok**. Same count, order, ids, key order and non-text fields; speaker tags and
  line breaks are unchanged.
* Audit (`outputs/eval_patient_doctor_encounters.anonymized.audit.jsonl`): 1,242 decisions, all
  explained, no original text.
* QI check: smallest class 3, 0 records below k; escalation levels 58 / 2 / 1 / 6.
* Runtime: 1.33 s/encounter, 1.63 GB peak including workers.

**Verified results** (`outputs/results.json`; local proxies, not Bosch scores). All values are eval
except canaries, which are injected into dev encounters.

| | Result |
|---|---|
| Name seeds leaked | 0 / 62 |
| Held-out adversarial recall | 0.786 [0.755–0.815], precision 0.938 |
| Canaries removed | 99.96 % of 2,312 |
| ICD-10 terms kept | 100 % |
| ICD-proxy F1 | Δ 0.000 |
| Negation cues kept | 1696 / 1696 |
| Correct unique linkage | 58.2 % → 3.0 % |
| Lab linkage at 3 % tolerance | 70.6 % (only `max_privacy` intervals: 11.8 %) |
| Unique on (age band, sex) | 40 % → 0 % |
| Unique on ICD code set | 97 % (inherent) |

**Reproducibility.** From a clean `git clone` and a new venv:
`pip install -r requirements.txt`, then `medanon prepare-models` (empty cache → downloaded), then
`medanon anonymize --offline` with the network blocked, then `medanon validate`. All succeeded and the
ML detector was active. Same `MEDANON_SECRET` → byte-identical output.

## 4. Remaining risks (no open P0)

* No human-annotated real ground truth yet: run `scripts/annotation.py export`, annotate the 25
  validation encounters locally, then `scripts/annotation.py score`.
* Held-out synthetic misses:
  * persons named like drugs with only an apposition cue: 0 %;
  * ordinary-word employers without a work cue ("i'm at amazon now"): 55 %;
  * occupations without a cue: 18 % (default profile);
  * relatives' ages phrased as "when she was sixty".
* The team repo is private but still holds the raw datasets and one real surname in old docs
  (§5.7).
* The QI check covers age band, sex, occupation category and ethnicity only. ICD code sets alone make
  most encounters unique, which is inherent to keeping the codes (see the uniqueness numbers).

## 5. Human actions before submission

1. Add every team member's full name and role to README.md and REPORT.md.
2. Trim REPORT.md to 3 pages and export it as PDF. The team's `main` also has an empty
   `report/report.tex`; pick one source.
3. Record the 2-minute video and prepare the Swisstransfer link.
4. Decide how to hand over the anonymized eval JSONL. It is git-ignored on purpose; delivery is by
   e-mail or link, or by committing that one file deliberately.
5. Send the mid-term evaluation **only after team approval**.
6. Run `python scripts/privacy_check_docs.py`; it must report `FAIL 0` (the `dev` tip does). Then
   test a clean checkout (`pip install -r requirements.txt`, `medanon prepare-models`).
7. **Publish the local branch `release`, not the history of `dev`.** State of the team repo,
   checked read-only on 2026-10-17:
   * The repo is **private**: the GitHub API returns 404 without authentication.
   * `main` (after PR #1) and `medanon` contain both raw datasets, from the initial commit
     `5bf721c`, and the session-1 README with one real surname. The gate reports FAIL 7.
   * Every `dev` commit except the latest has the same old docs (FAIL 7). None ever contained the
     data.

   `release` is a single commit with the `dev` tip tree: gate FAIL 0, no data in its history.
   Publish it as a **new repository**, or force-push it to `main` and delete `medanon`. GitHub keeps
   PR #1's commits reachable after a force-push, so only a new repository (or GitHub Support)
   removes the data from the old one. Until then, keep the old repo private.
8. Optional: human annotation (§4) and choosing `max_privacy` vs `default` after the Bosch feedback.

## 6. Update 2026-10-17: maximum privacy is the default

At the team's request the default (`config.py`, `configs/default.yaml`) is now the former
`max_privacy` profile: OpenMed PII SuperClinical over dialogue and note, lab values as category
intervals. The former default is `configs/standard.yaml`; the README/REPORT result tables were
measured with it.
* The manifest now records the extra detectors (revision, weights SHA-256). `--offline` and
  `--no-ml` also apply to them, and `prepare-models` caches them.
* Four bugs, all made visible by the new default, are fixed, each with a regression test (100 tests):
  * OpenMed spans started with the preceding space ("works as a[OCCUPATION]");
  * a merged span lost the occupation category of the rule span;
  * the QI step crashed when occupations with and without a category met in one record;
  * the lab rule matched a spoken number inside a word ("alone" → "al[0.6-1.2]"). With
    perturbation that value had stayed unchanged, so only interval generalisation exposed it.
* Eval with the new default: ~15 s/encounter, 3.1 GB, validate ok, name seeds 0 / 62, ICD terms
  100 %. Every replacement made by `standard` is still covered.
* `outputs/submission/` is still the `standard` output until it is regenerated.
