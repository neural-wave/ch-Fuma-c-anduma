# Third-party components and data

MedAnon source snapshot of release commit `e6c3ce7`. The project licence is still to be chosen by the team; this
file credits the third-party software, models and data the release depends on. Each item stays under its own licence.

| Component | Use | Licence |
|---|---|---|
| spaCy, `en_core_web_lg` 3.8.0 (Explosion) | tokenisation, NER | MIT |
| Microsoft Presidio (analyzer) | pattern recognisers | MIT |
| Hugging Face Transformers | model runtime | Apache-2.0 |
| PyTorch | model runtime | BSD-3-Clause |
| `dslim/bert-base-NER-uncased` | dialogue NER (downloaded by `medanon prepare-models`) | MIT (see model card) |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | clinical PII NER (downloaded by `medanon prepare-models`) | Apache-2.0 (see model card) |
| PyYAML, psutil, scikit-learn | config, telemetry, metrics | MIT / BSD-3-Clause / BSD-3-Clause |

Lexicons in `src/medanon/resources/` (built by `scripts/build_resources.py`):

* `surnames.tsv`, `first_names.tsv`: US Census Bureau (2010 surnames, 1990 first names) and SSA baby-name data.
  US government work, public domain.
* `drugs.txt`: FDA NDC product directory. US government work, public domain.
* `icd_vocab.tsv`, `eponyms.tsv`: ICD-10-CM through `simple-icd-10-cm`. ICD-10-CM is a CDC/NCHS publication in the
  public domain.
* `common_words.tsv`: word frequencies from `wordfreq` (Robyn Speer). The wordfreq data is CC BY-SA 4.0, so this
  file is shared under the same licence, with attribution to wordfreq and its sources.
* `clinical_terms.txt`, `eponyms_extra.tsv`, `occupations.tsv`, `places.tsv`: curated by the team.

Check each model card and licence before redistributing model weights. The weights are not included in this snapshot.

Not included: Git history, the challenge datasets (training and evaluation encounters), any raw or anonymized
patient records, gold annotations, validation-subset ID lists, identity databases, research and benchmark code,
caches and virtual environments. Test fixtures are fictional.
