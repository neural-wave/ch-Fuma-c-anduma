# MedAnon: context-aware anonymization of doctor–patient encounters

_Neural Wave Hackathon 2026 — Bosch challenge. Team: TODO (names and roles)._

## 1. Problem and data

Each encounter has a lowercase ASR **dialogue** (`[doctor] hi , mr. kowalczyk ...`), a cased
physician **note** and ICD-10 codes that must stay unchanged. Evaluation rewards removing private
information *and* keeping what an ICD-10 coder and a concept extractor need, plus originality and
scalability. A quick look at the structure (aggregate statistics only; we never printed patient text)
drove the design: the dialogue has **zero uppercase characters**, so capitalization-based NER
misses names there; patient names, DOBs and MRNs are dictated at the start of many dialogues
(`<name> , date of birth 3/14/1958 , mrn 4-4-7-1-9-2`); numbers are often spoken ("eight point
three"); and the hardest utility trap is the medical vocabulary that looks like names
(eponyms such as Parkinson, Bell, Graves, Baker; drug brands such as Lasix or Allegra).

## 2. Approach

**Pipeline.** `detect → clinical guard → encounter registry & propagation → overlap resolution →
risk-adaptive transformation → dataset-level k-anonymity → explanation log`.

**Detection (high recall, cheap first).**
- *Rules tuned for speech:* titles, vocatives ("hi , rosalind ."), relation slots ("my wife , jenny"),
  dictation openers, header fields, ages incl. spoken numbers and "how old are you? — 62",
  DOB/date formats seen in the data, glued (`mrnq27631`) and digit-by-digit MRNs, phones/e-mails,
  institutions ("ohio state university"), employers ("work as a chef at target"), occupations
  with a cue, gazetteer places gated by locative cues.
- *Name-dominance ratio:* for a token *w*, P(name usage)/P(any usage), estimated from Census/SSA
  name frequencies and `wordfreq` English frequencies. It separates "rosalind"/"marisol" (≈1) from
  "will"/"hope"/"may" (<0.1) without hand-made lists, and sets how much context a token needs.
- *spaCy via Presidio:* `en_core_web_lg` NER on the note and on a **note-guided truecased view**
  of the dialogue (proper nouns written capitalized in the note are capitalized in the dialogue;
  same length, so offsets map 1:1), plus Presidio's validated pattern recognizers.
- *Transformer cascade:* an uncased BERT NER (`dslim/bert-base-NER-uncased`, trained on lowercased
  text) runs only on dialogue turns that still contain an unexplained name-like or rare token, and
  its spans are accepted only with lexical support (see §3 for how we chose it).

**Clinical-context guard.** Every PERSON/ORG/LOCATION/NATIONALITY candidate is checked against
(i) 230 eponyms mined from ICD-10-CM (capitalized non-initial words in its 98k descriptions) plus a
curated list, each with the head nouns that make it medical (*disease, palsy, cyst, fracture...*),
(ii) an FDA-NDC drug lexicon, (iii) curated clinical abbreviations, (iv) the record's own ICD-10
descriptions. Syntactic person evidence (title, relation slot) overrides lexical medical evidence,
so "Mr. Parkinson" is anonymized while "Parkinson's disease" stays.

**Encounter registry.** Person mentions are clustered by name-token overlap, assigned roles by
evidence voting (title "dr." → clinician; "my wife" → relative; note subject → patient; who is
addressed in which speaker turn) and every surface form is propagated to both fields with
ambiguity gating (a name that is also a common word needs local context). Places and organizations
found anywhere are propagated the same way. This is what makes note and dialogue consistent.

**Transformations.** Names → `[PATIENT_NAME]`/`[CLINICIAN_NAME]`/`[RELATIVE_NAME]`/`[PET_NAME]`
(not `[PATIENT]`, which would collide with the `[patient]` speaker tag); ages → 10-year bands;
dates, DOB, IDs, contacts → placeholders; occupations → category; countries → world region;
Lab values (HbA1c, glucose, lipids, creatinine/eGFR, TSH, PSA, EF, …) get a perturbation of at most
3 %, chosen so the value **stays in the same guideline category** (e.g. HbA1c thresholds
5.7/6.5/7/8/9/10). Values that cannot move within 3 % at their precision, and values inside reported
ranges, are kept. The perturbation is keyed by a secret (not by public data), so it cannot be
recomputed; note and dialogue, digits or spoken words, stay consistent. In `max_privacy`, lab values
become category intervals instead.

**Quasi-identifier step.** After transformation, a record's published demographic QIs (age band,
sex, occupation category, ethnicity) are counted across the dataset. Records in classes smaller than
k=3 get wider age bands and suppressed occupation, then coarse age, then age suppressed. A post-hoc
check reports the final class sizes. This covers the declared QIs only, not free text, dates, labs
or diagnoses, so it is not formal k-anonymity.

**Explainability and traceability.** Every decision, including *kept* look-alikes (with a reason
code), is logged with category, action, replacement, risk class, detectors, confidence and a
sentence of evidence, never with the original text. Every run writes a manifest: code, config and
model hashes, the effective detector status, and the input and output hashes. The configured model is
required; there is no silent fallback.

## 3. Results

No gold annotations exist, so we built four complementary measurements and report them as
*proxies*, never as Bosch scores. The 67-record eval split was held out from all tuning.

1. **Synthetic gold** — generated encounters in ACI dialogue style, discharge-summary style and
   cased-dialogue style, with exact PHI spans and 1,460 medical "traps" (eponyms, drug brands,
   place-named diseases, patients *named* Parkinson who *have* Parkinson's).
2. **Canary injection** — 2,312 synthetic identifiers inserted into real dev encounters (new
   dialogue turns, sentences inside note sections): recall in genuine clinical context, without
   anyone reading the records.
3. **Real-data proxies** — an *independent* detector (vanilla Presidio) on the originals,
   high-precision name seeds, a medical-concept extractor applied to original and output, and a
   TF-IDF ICD-10 coder trained on dev and tested on eval original vs. anonymized.
4. **Attacks** — linkage with background knowledge (correct vs. wrong unique matches separated),
   exact and tolerant lab-value linkage, perturbation inversion, residual-name harvesting.
5. **Held-out adversarial set**, built after development from disjoint lexicons and phrasings:
   40 encounters, 720 gold identifiers, 14 clinical traps each.

| | Proxy | Result (`standard` profile) |
|---|---|---|
| Privacy | patient-name seeds still present anywhere in the output (eval / dev) | **0 / 62** · 1 / 120 |
| Privacy | held-out adversarial recall · precision | **0.786** (95 % CI 0.755–0.815) · 0.938 |
| Privacy | canary identifiers removed (2,312, real dev context) | **99.96 %** (fast profile 98.3 %) |
| Privacy | vanilla-Presidio items removed, calendar-anchored dates only (eval / dev) | 87.1 % / 87.9 %* |
| Utility | ICD-10 terms retained · negation cues kept · numbers outside replacements changed | **100 % · 1696/1696 · 0** |
| Utility | ICD-10 TF-IDF coder micro-F1, original → anonymized (eval) | 0.384 → **0.384** |
| Utility | adversarial clinical traps intact | 0.929 (family-history ages "when she was sixty" are banded) |
| Attack | **correct** unique re-identification, linkage with age/sex/job/city known | **58.2 % → 3.0 %** (wrong-unique 0 %) |
| Attack | unique lab linkage: exact match · 3 %-tolerant | 76 % → 18 % · **71 %** (category intervals: 12 %) |
| Attack | records unique on (age band, sex) · on the ICD-10 code set | 40 % → **0 %** · 97 % (inherent) |
| Other | decisions explained · s/encounter (1 CPU core) · RAM · GPU | 100 % · 1.3 s · 1.6 GB · none |
| Bonus | outputs identical without ICD-10 context | 66 / 67 |

\*the survivors on dev are mostly the reference detector's own false positives (not gold-verified).

**Should a transformer be part of the pipeline?** We benchmarked the two GLiNER PII models the team
suggested and an uncased BERT NER, applied to the whole dialogue or only to "uncertain" turns
(cascade):

| Configuration | Synth. recall | Synth. precision | Canary | Cue-less names | Extra real edits* | s/enc. | RAM |
|---|---|---|---|---|---|---|---|
| rules + spaCy | 0.995 | 0.954 | 0.986 | 0.865 | – | 0.42 | 1.2 GB |
| + GLiNER-PII (full) | 0.996 | 0.949 | 0.998 | 0.981 | +19 | 3.73 | 3.0 GB |
| + GLiNER2-PII (full) | 0.997 | 0.951 | 1.000 | 1.000 | +61 | 5.21 | 2.7 GB |
| + BERT-uncased (full) | 0.997 | 0.954 | 1.000 | 1.000 | +8 | 2.31 | 1.7 GB |
| + BERT-uncased (cascade) | 0.997 | 0.954 | 1.000 | 1.000 | +7 | 1.42 | 1.6 GB |

\*additional replacements on 60 real dev encounters. Plugged in naively, GLiNER/GLiNER2 made +610 /
+2,053 extra edits — mostly relabelling the `[doctor]`/`[patient]` speaker tags and ordinary words —
and synthetic precision fell to 0.903 / 0.849. After adding a speaker-tag guard and an evidence
filter (model spans need lexical support), the uncased BERT cascade gave perfect canary recall
with no precision loss; on all 140 dev encounters it adds four spans (one probable name, three
organisation-like words). It is the default; `configs/fast.yaml` drops it for 3× throughput.

**Newer models** (`MODEL_EXPERIMENTS.md`, same held-out set and scorer). Healthcare-Brain-NER gave
+0.8 points, not significant. Qwen3.5-2B (4-bit GGUF, grammar-constrained JSON whose spans are
validated verbatim) produced 99.4 % valid JSON with 1.7 % hallucinated spans rejected, but only +1.1
points at 57 s per encounter on CPU. OpenMed PII SuperClinical raised recall **0.786 → 0.886** with
non-overlapping CIs and no precision loss, mostly through occupations without a cue. It costs ~12×
CPU time and changed only 2 spans in 25 real encounters. It is part of the default since 2026-10-17
(maximum privacy); `configs/standard.yaml` is the faster profile measured in the table above.

**What the audit of "leaks" taught us.** The independent detector's surviving items on dev were
overwhelmingly its own false positives — body parts ("mouth", "throat"), drug brands, the ASR
artefact "wan na", eponymous exam signs ("positive Spurling's sign") and the ambient-AI assistant's
name — which we leave intact on purpose. The few true misses each revealed a missing pattern
(a closing vocative without comma, a dictated "X is a 62-year-old" in the dialogue, a surname after
a detected first name), which we then fixed generically and re-checked on dev.

## 4. What did not work / lessons

- spaCy NER directly on the lowercase dialogue found almost no names; truecasing from the note and
  speech-specific rules were needed.
- The raw FDA product file made "Arizona", "Johnson", "Timothy" and "Kroger" look like drugs
  (allergen extracts, store brands); without cleaning, the guard protected real identifiers.
- Propagating every organization string turned "ups" (employer) into a hit inside "ups and downs";
  common-word entities now propagate only into entity-like slots.
- Surrogate names and date shifting were rejected: if the organizers' privacy check runs a PII
  detector on our output, realistic fake names/dates would be counted as leaks; placeholders are
  safe under both string-matching and detector-based evaluation.
- Bare `[PATIENT]` placeholders collide with the `[patient]` speaker tags.
- An independent review found real defects. We fixed them and added regression tests (99 tests in
  total):
  - spoken-digit MRNs and phone numbers were left in clear;
  - the audit stored original text of kept candidates;
  - the model fallback was silent;
  - lab values could move by more than 3 % (TSH 0.1 → 0.2);
  - the perturbation seed was public and therefore invertible;
  - the linkage metric counted zero-candidate matches as successes.
- Our first, template-based synthetic benchmark was too easy (recall 0.99). An independent adversarial
  set exposed failures with unseen phrasings and diacritic names (recall 0.67 at first). We fixed only
  generic causes and report final numbers on a second held-out bank.
- A small perturbation does not stop an attacker who matches lab values approximately (71 % unique
  links at 3 % tolerance). Only category intervals do.

## 5. Use of AI tools

The implementation and experiments were developed with an AI coding assistant (Claude Code) under
the team's direction. A second AI assistant performed an independent code review, whose findings were
verified and fixed. Patient data were never sent to either: they only saw aggregate statistics and
masked diagnostics, and all debugging used synthetic encounters. Human review of the real-data
candidate spans (`outputs/review/`) and annotation (`scripts/annotation.py`) are done locally by
the team.
