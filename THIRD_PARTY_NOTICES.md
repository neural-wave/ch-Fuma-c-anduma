# Third-party notices

MedAnon's own code is released under the MIT licence (see [`LICENSE`](LICENSE)). The third-party software, models
and data below keep their own licences, given as stated by each project or model card; check the upstream source
before redistributing.

## Runtime software (installed by `requirements.txt`, not vendored)

| Component | Licence |
|---|---|
| spaCy and the `en_core_web_lg` pipeline | MIT |
| Microsoft Presidio (analyzer) | MIT |
| Hugging Face Transformers | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| PyYAML | MIT |
| psutil | BSD-3-Clause |

## Pretrained models (downloaded once by `medanon prepare-models`, not vendored)

| Model | Use | Licence |
|---|---|---|
| `dslim/bert-base-NER-uncased` | NER cascade on the dialogue | MIT, per its model card |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | PII detection on note and dialogue | Apache-2.0, per its model card |

No model was fine-tuned or modified.

## Lexicons shipped in `src/medanon/resources/`

Compiled by `scripts/build_resources.py` from:

| Source | File(s) | Licence |
|---|---|---|
| US Census 2010 surnames; US Census 1990 first names; SSA baby names (via the hadley/data-baby-names mirror) | `surnames.tsv`, `first_names.tsv` | US government work, public domain |
| FDA NDC product directory | `drugs.txt` | US government work, public domain |
| ICD-10-CM (CDC/CMS), read with the `simple-icd-10-cm` package at build time | `icd_vocab.tsv`, `eponyms.tsv` | US government work, public domain |
| `wordfreq` English word frequencies | `common_words.tsv` | CC BY-SA 4.0 (wordfreq data); this derived file is shared under the same terms |

`clinical_terms.txt`, `eponyms_extra.tsv`, `occupations.tsv` and `places.tsv` are hand-curated lists maintained in
this repository (clinical terms, eponym supplement, occupation-category map, place gazetteer); see their headers.

## Used only in comparison studies or research code (not in any release profile)

- GLiNER and GLiNER2 PII models, Healthcare-Brain-NER, Qwen3.5-2B (GGUF) with llama.cpp: detector comparison
  (`scripts/compare_detectors.py`, `scripts/model_experiments.py`, `src/medanon/llm.py`). Licences per their model
  cards and repositories.
- Synthea (synthetic patient generator) and Splink (record linkage): `research/reid/`.
- A local LLM served by Ollama and the packages in `attack/requirements.txt`: the teammate's attacker in `attack/`.

## Data

The Bosch challenge data (Yuan, Moy et al., EMNLP 2025 Industry Track; ACI-Bench-derived encounters) are not
included and must not be added to this repository.
