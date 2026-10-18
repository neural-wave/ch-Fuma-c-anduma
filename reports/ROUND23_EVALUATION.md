# MedAnon – synthetic development rounds 2 and 3 (evaluation protocol and results)

Data: 2 × 20 **synthetic development** encounters (`SYN26R2_001–020`, `SYN26R3_001–020`) with author-labelled
reference spans (not independently adjudicated). The validation encounters of both rounds were **not** present
and were not used. These 40 records are now development data: the numbers below are not an estimate of
performance on the Bosch hidden set.

## Provenance

| | Baseline B0 | Candidate B2 |
|---|---|---|
| code | commit `096ff01`, code_sha256 `882d7098f858` | working tree on top of `096ff01`, code_sha256 `39755f6242d3` |
| config | default (`d85e6375efdd`), CPU | default + `--device cuda` (`de89e012899e`); candidate age policy `configs/age_bands.yaml` (`75519c86dcae`) |
| outputs | `run2/out_v5`, `run3/out_v6` (provided; manifests verified) | regenerated on the same inputs (GPU) |

Inputs: round 2 `5e0bceb3f2d5…`, round 3 `68483988969a…`; gold: `739099cfd8ea…`, `f19b3c7e279c…`.
Detectors in both: rules, spaCy `en_core_web_lg` 3.8.0 + Presidio, `dslim/bert-base-NER-uncased`@301540b,
OpenMed PII SuperClinical 434M@df7af99, MedicalGuard. Structure (IDs, `dataset`, `icd10_codes`, line counts) intact
in 40/40 records for every version.

## Metric definitions

* **Raw recall**: gold privacy entities (all labels except `CLINICAL`) whose alphanumeric characters were all
  transformed. Ages / years published by policy count as **misses**.
* **Policy-compliant recall**: removed + deliberately published by policy (reported separately).
* **Precision**: transformed spans overlapping a gold privacy entity / all transformed spans.
* **Strict token removal** (proxy, *not* Bosch's metric): an entity counts as removed only if none of its tokens
  is still in the output field — so `[DOB] 1985` or a kept age is *not* removed. Over-strict for common words.
* **Clinical spans intact**: `CLINICAL` reference spans (negations, doses, lab results, exam, management,
  family-history onset ages) left completely unchanged.

## Results (same 20 records per round)

| Metric | R2 B0 | R2 B2 | R2 B2+age | R3 B0 | R3 B2 | R3 B2+age |
|---|---|---|---|---|---|---|
| Raw privacy recall | 0.807 | 0.851 | **0.970** | 0.765 | 0.794 | **0.915** |
| Policy-compliant recall | 0.941 | 0.985 | 0.985 | 0.900 | 0.929 | 0.929 |
| Precision | 0.899 | **0.988** | 0.989 | 0.899 | **0.988** | 0.990 |
| Strict token removal (proxy) | 0.658 | 0.706 | **0.803** | 0.545 | 0.573 | **0.762** |
| PERSON recall | 0.885 | 1.000 | 1.000 | 0.933 | 1.000 | 1.000 |
| Partially removed names | 9 | 0 | 0 | 6 | 0 | 0 |
| Transformations outside gold (FP) | 26 | 3 | 3 | 27 | 3 | 3 |
| Clinical spans wrongly replaced | 0 / 4 | 0 / 4 | 0 / 4 | 17 / 672 | 1 / 672 | 1 / 672 |
| Clinical spans not verbatim (incl. lab-interval policy) | 0 / 4 | 0 / 4 | 0 / 4 | 29 / 672 | 13 / 672 | 13 / 672 |
| Records below k=3 (declared QIs) | 20 | 20 | 8 | 20 | 20 | 13 |
| s / encounter (GPU) · peak VRAM | 15.2 (CPU) | 0.70 · 2.6 GB | 0.67 · 2.6 GB | 14.3 (CPU) | 0.68 · 2.6 GB | 0.68 · 2.6 GB |

"Not verbatim" includes lab values deliberately generalised to their clinical interval (declared policy).
"Records with a direct identifier left" in round 3 (0.45 → 0.35) is entirely
the cue word `MRN` kept before `[ID]` (the reference span includes it); no ID value survives.
Run 1 (25 synthetic records) used as a non-regression check: identical issue counts, no identifier newly visible.

## What explained the observed weaknesses (confirmed on development data)

1. **Published ages and years dominate the gap.** Ages are 14 % of the reference entities; with the exact-age
   policy 75 of 80 were published, and DOB/visit dates kept their year. Under the strict token proxy AGE ≈ 0,
   DATE = 0, DOB ≈ 0.1. This is *consistent* with Bosch's 0.59 if their metric is token/value-based, but it does
   not prove it.
2. **Detection gaps**: letter-prefixed record numbers; ordinary-word / medical-word surnames after a first name;
   a business named with "the workplace is called"; generic workplace descriptions mistaken for employers.
3. **Clinical false positives**: derived medical words (supra-/sub- prefixes, -al/-ic adjectives), named
   instruments (… Scale/score), hyphenated eponyms split by sub-word NER, units without a preceding number,
   mixed-case analytes with values, drug combinations, pain ranges read as dates, staff role words as
   occupations.

## Code changes (general, with positive tests and counterexamples in `tests/test_round23_regressions.py`)

Rules (ID pattern; capitalised census surname after a first name unless a date/medical head follows; single-
separator dates; `born <date>` = DOB; "practice"/"medical group" heads; business subjects; generic kin terms in
family-onset exclusion) · pipeline (hyphen-compound span extension; unit and analyte-value guards; statistical
first name + surname; staff role words; all-caps headers) · overlap resolution (strong rule evidence decides the
label) · lexicon (clinical prefix/suffix derivations, drug compounds; no new name-lexicon collisions: 126 → 126)
· guard (plural / "testing" heads, eponyms reported as findings, named instruments) · candidate age policy.

## Candidate age policy (`configs/age_bands.yaml`, NOT the default)

Exact age 0–17, `[18-19]`, ten-year bands to `[80-89]`, `90+`; adult birth year removed (paediatric kept);
family-history onset ages untouched. Measured: AGE strict removal 0.05→0.60 / 0→0.75, DOB 0.14→0.43 / 0.08→0.92,
records below k 20→8 / 20→13, fewer occupations suppressed by the QI step; no clinical span damaged by it.
Clinical cost (not measured by these references): within-decade thresholds (e.g. 35, 45, 65, 75) are no longer
determinable for adults; patients' own past-event ages are banded too.

## Unresolved

* Occupation spans partially removed by reference convention ("retired …", "school …") — 4 / 8 mentions.
* One lab passage and one all-caps header still mis-handled in round 3; ID cue-word convention.
* Remaining QI uniqueness is driven by ICD-10 codes and retained clinical detail, not only age; no formal k-anonymity.
* Bosch's 0.5918 cannot be reproduced locally; its definition and entity mix are unknown.

## Reproduce

```
medanon anonymize --offline --device cuda -i <dev20.jsonl> -o <out.jsonl> [--config configs/age_bands.yaml]
python scripts/score_gold.py <dev20.jsonl> <out.jsonl> <dev20_gold.jsonl> --json report.json
python scripts/compare_run25.py <run1 input> <before out> <after out>
```
