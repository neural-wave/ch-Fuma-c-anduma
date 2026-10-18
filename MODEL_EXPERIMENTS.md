# Model experiments — do newer local models improve MedAnon?

**Question.** Can a specialised clinical NER model or a small instruction-tuned LLM detect privacy
information that the current pipeline misses, without damaging clinical content and within the
Bosch constraints (local inference, ≤ 4 GB VRAM)?

**Answer.** On held-out synthetic hard cases, only **OpenMed PII SuperClinical** produced a
statistically clear gain: recall 0.786 → 0.886, with no precision or clinical-trap loss. The gain comes
mostly from occupations without an explicit cue. On 25 real development encounters it added only
2 replacements, at ~12× the CPU cost. Healthcare-Brain-NER and Qwen3.5-2B gave no significant gain.
**Decision:** the default pipeline stays unchanged, and OpenMed is offered as an opt-in
`configs/max_privacy.yaml` profile.

## 1. Candidates (all Apache-2.0, run locally from the Hugging Face cache)

| | Model (revision) | Type / size | Labels used | Backend |
|---|---|---|---|---|
| Baseline | rules + spaCy `en_core_web_lg` 3.8.0 + `dslim/bert-base-NER-uncased` (301540b) cascade | BERT-base 110M | PER/LOC/ORG | transformers 5.16.1, CPU |
| A | `genzeonplatform/healthcare-brain-ner` (2873f61) | BERT token classifier, 431 MB | PATIENT/DOCTOR_NAME, HOSPITAL, CITY/STATE/COUNTRY/ZIP/ADDRESS, MRN/ID/SSN, PHONE/FAX/EMAIL/URL | transformers, CPU |
| B | `Qwen/Qwen3.5-2B` → `unsloth/Qwen3.5-2B-GGUF` Q4_K_M (f6d5376), 1.28 GB | instruction LLM, structured extraction | PERSON, LOCATION, ORGANIZATION, OCCUPATION, ID, CONTACT (+ MEDICAL outlet) | llama-cpp-python 0.3.36, CPU |
| C | `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` (df7af99) | DeBERTa-v3 token classifier, 1.74 GB | first/last name, company, city/state/country/address/postcode, MRN & other IDs, phone/fax/email/url, occupation | transformers, CPU |

Model output never edits text directly. Each candidate's spans go through the same evidence filter
and clinical guard as the baseline's ML spans: persons need name evidence, places and organisations
need a distinctive non-medical word, and occupations must not be generic clinical roles. Ages and
generic dates from the models are ignored; the rules handle those with clinical safeguards. Qwen
returns grammar-constrained JSON `{text, type}` items. Python keeps an item only if it occurs
verbatim in the passage; anything typed MEDICAL is discarded.

The transformers bf16 build of Qwen3.5-2B needed 66–174 s per short passage on this CPU
(≈0.2 tokens/s, bf16 is emulated). It was replaced by the 4-bit GGUF, which uses 1.2 GB RAM and loads
in 2 s.

## 2. Setup

* **Same code for every row.** Harness commit `5f4e5c0`; the baseline's detection behaviour is
  identical to baseline v1 (`e2f1dd7`). Each candidate is added as one `extra_detectors` entry in
  `scripts/model_experiments.py`. Every configuration runs in a fresh
  process with the same seeds, records, thresholds and scorer
  (`src/medanon/evaluation/scoring.py`).
* **Held-out adversarial synthetic set.** `adversarial.generate(bank="test")`: 40 encounters,
  720 exact-gold identifiers, 14 clinical traps per encounter. Its lexicons and phrasings are
  disjoint from everything used during development, and it was generated after the rule fixes. It
  covers names with diacritics or out-of-vocabulary names, common-word names, persons named like
  drugs, eponym surnames, spoken-digit IDs, identifiers at the end of very long turns, ordinary-word
  employers and rare occupations. Traps cover negation, family history, dosage, lab values, eponyms,
  temporal expressions and diagnoses.
* **Real data, no gold.** A fixed validation subset of 25 development encounters
  (`configs/validation_subset.json`, stratified by dataset style). For each candidate we count
  replacements added or removed versus the baseline, reported as privacy-safe categories. The
  surfaces themselves go only to `outputs/review/<config>_added_spans.jsonl` for local human review.
  Utility integrity checks: negation-cue counts and every number outside replaced spans.
* **Human annotation.** `scripts/annotation.py export` produces blind Label Studio tasks for the 25
  encounters; `scripts/annotation.py score` scores every configuration once they are annotated.
  **This has not been done yet**, so there are no human-annotated numbers below.
* **Hardware.** 12-core laptop CPU, 16 GB RAM, no GPU. Peak RAM covers the whole process (all models);
  VRAM was not used.

## 3. Results

Recall and precision are entity-level, with Wilson 95 % intervals. "Real +" counts replacements
added versus the baseline on the real validation records (none were removed in any configuration).

**Full comparison** (40 adversarial encounters, 720 identifiers; 25 real encounters):

| Configuration | Recall | Precision | F1 | Traps | Real + | s/enc synthetic | s/enc real | Peak RAM |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.786 [0.755–0.815] | 0.938 [0.916–0.954] | 0.855 | 0.929 | – | 0.79 | 1.36 | 1.6 GB |
| + A Healthcare-Brain | 0.794 [0.763–0.822] | 0.935 [0.912–0.952] | 0.859 | 0.929 | 5 (persons) | 1.50 | 3.44 | 2.0 GB |
| + C OpenMed, all text | **0.886 [0.861–0.907]** | 0.944 [0.924–0.959] | **0.914** | 0.929 | 2 (occupations) | 6.87 | 17.2 | 2.9 GB |
| + C OpenMed, cascade | 0.824 [0.794–0.850] | 0.940 [0.919–0.956] | 0.878 | 0.929 | 1 | 2.54 | 8.02 | 3.0 GB |

**Qwen comparison**, run on a reduced but identical sample because of its cost (20 adversarial
encounters, 360 identifiers; 10 real encounters):

| Configuration | Recall | Precision | F1 | Real + | s/enc synthetic | s/enc real | Peak RAM |
|---|---|---|---|---|---|---|---|
| Baseline | 0.769 [0.723–0.810] | 0.936 | 0.845 | – | 0.69 | 1.25 | 1.6 GB |
| + C OpenMed, all text | 0.883 [0.846–0.913] | 0.944 | 0.913 | 1 | 5.12 | 13.3 | 3.0 GB |
| + B Qwen3.5-2B Q4, cascade | 0.781 [0.735–0.820] | 0.937 | 0.852 | 4 | 19.3 | 57.3 | 3.7 GB |

Recall by difficulty on the full sample:

| | baseline | A | C | C cascade |
|---|---|---|---|---|
| occupation without explicit cue | 0.175 | 0.175 | **1.000** | 0.438 |
| employer that is an ordinary word | 0.550 | 0.550 | 0.588 | 0.588 |
| name with diacritics / out of vocabulary | 0.925 | 0.975 | 0.950 | 0.950 |
| person named like a drug ("celexa, my coworker") | 0.000 | 0.000 | 0.000 | 0.000 |
| end of a very long turn | 0.925 | 0.925 | 0.925 | 0.925 |
| spoken-digit IDs, ages, eponym surnames, unseen cities, common-word names | 1.000 | 1.000 | 1.000 | 1.000 |

**Utility.** In every configuration, negation cues were identical before and after (596 / 596), no
number outside a replaced span changed, and trap preservation was the same (0.929). The only trap
type below 1.0 everywhere is family history (0.5): "my mom had a stroke when she was sixty" gets
its age banded. This is a known baseline gap that no model fixes.

**Qwen reliability** (157 calls):

* valid JSON 99.4 % (grammar-constrained);
* items not found verbatim in the passage (hallucinated or paraphrased): 1 / 60 (1.7 %), rejected;
* 16 items labelled MEDICAL and discarded;
* 47 spans accepted before filtering;
* 5.9 s per call, 2.2 generated tokens/s.

## 4. Failure modes

* **A, Healthcare-Brain.** Its label set matches PHI well, but it adds little beyond spaCy and the
  uncased BERT: a few names in notes, nothing on occupations or employers. It probably saw mostly cased
  synthetic PHI during training. Its 44 downloads mean little external validation.
* **B, Qwen3.5-2B.** Strong contextual reading: it found "forklift mechanic", "amazon" and "sunny"
  in probes and returned `[]` on purely clinical lines. But in the cascade it only sees turns that
  contain an unexplained name-like token, so most occupation and employer sentences never reach it.
  It also files person names that are drug names ("celexa") as MEDICAL. Covering every line would
  multiply the already 42× CPU cost. Prompt sensitivity: without a MEDICAL outlet the grammar forces a
  PII type, and it labelled "ativan" as an ORGANIZATION. The guard catches that, but it shows that
  constrained decoding needs an abstain or other option.
* **C, OpenMed.** Its label set includes occupations, so it finds jobs without any cue. With the
  name-based trigger, the cascade misses job-only sentences: the gain drops from +10 to +4 points.
  On real data it changed 2 spans in 25 encounters. One of them is a common word in a note, which may
  be a false positive (left for human review).
* **None of the models** resolves a person named like a drug when the only cue is a trailing
  apposition ("celexa , my coworker , told me").

## 5. Decision

| Criterion | A | B | C |
|---|---|---|---|
| Significant privacy-recall gain | no (+0.8 pt, CIs overlap) | no (+1.1 pt) | **yes (+10 pt)** |
| Precision maintained | yes | yes | yes |
| Clinical utility preserved | yes | yes | yes |
| Runtime / memory acceptable | 2.5×, 2 GB | **no** on CPU (42×, 3.7 GB) | 12×, 3 GB (OK on GPU) |
| Worth the complexity | no | no | as an option |

* **Default** = baseline v1 (`configs/default.yaml`). It is the simplest configuration and the best
  per unit of compute, and on real encounters the candidates changed almost nothing.
* **`configs/max_privacy.yaml`** = default + OpenMed over all text. Use it when privacy outweighs
  throughput, or on a GPU, where DeBERTa-large fits easily in 4 GB of VRAM.
* **Not integrated:** Healthcare-Brain and Qwen. The Qwen detector, with grammar-constrained JSON
  output, verbatim span validation and a MEDICAL outlet, stays in `src/medanon/llm.py` as a documented
  research path. Running it on every line is only realistic on a GPU.

## 6. Limitations

* There is no human-annotated real ground truth yet; the blind annotation workflow is ready.
  Synthetic sets, however adversarial, are written by us. Their category mix sets the size of the
  gains: occupations are 11 % of synthetic gold but rare in the real encounters.
* The Qwen run used 20 + 10 encounters, so its intervals are wide. An effect of a few points cannot
  be excluded, but one of OpenMed's size can.
* All timings are single-process CPU measurements on one laptop. GPU timings were not measured.
* The adversarial development bank was used to find failure classes and fix them generically. Only
  the test bank, built afterwards, is used here; even so, its categories mirror the ones we already
  knew to probe.
