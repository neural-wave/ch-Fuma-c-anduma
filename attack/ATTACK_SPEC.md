# Re-identification attack — implementation spec

> Privacy note (2026-10-18): real encounter IDs are replaced by neutral labels (R1–R10, T1–T3), and record-specific names, places and verbatim quotes by paraphrases. Per-record files keyed by real IDs stay local.

## Context

This repo contains an anonymization pipeline for doctor–patient encounters (Bosch Neural Wave hackathon). Your job is to build the **optional bonus task "Adversarial Privacy Attacks"**: a separate attack tool that takes an anonymized JSONL file, measures how identifiable each patient still is, and tries to recover masked values.

**Hard constraints**
- The attack is **logically independent** from the anonymization pipeline. Put everything in a new top-level folder `attack/`. Do **not** modify, import from, or depend on the pipeline code. Own `attack/requirements.txt`, own `attack/README.md`.
- Only use the local LLM endpoint described below. No cloud APIs.
- Do not commit `attack/cache/` or `attack/results/` (add them to `.gitignore`).

## Data

- Anonymized input (default): `data/output_eval_patient_doctor_encounters.jsonl`
- Original input (optional, default): `data/eval_patient_doctor_encounters.jsonl`
- If these paths don't exist, search the repo for the files and ask before guessing.

Each line: `{"encounter_id", "dataset", "dialogue", "note", "icd10_codes": [{"code","description"}]}`. Join records between files by `encounter_id`.

Placeholders produced by the pipeline (treat as **unknown values**, never as extracted values):
`[PATIENT_NAME]`, `[CLINICIAN_NAME]`, `[CLINICIAN_NAME_1]`, `[PERSON]`, `[PERSON_2]`, `[RELATIVE_NAME]`, `[DOB] 1982`, `[DATE]`, `[DATE] 2021`, `[MONTH]`, `[WEEKDAY]`, `[HOLIDAY]`, `[ID]`, `[US_STATE]`, `[LOCATION]`, `[LOCATION: Middle East]`, `[ORGANIZATION]`, `[HEALTHCARE_FACILITY]`, `[EMPLOYER]`, `[OCCUPATION]`, `[ETHNICITY]`, `[NATIONALITY]`, `[ADDRESS]`.
Generalized lab values look like `[40-50]`, `[<5.7]`, `[>=240]`; generalized ages look like `90+`.
Placeholder regex: `\[[A-Z][A-Z_]*(?:_\d+)?(?::[^\]]*)?\]`.

## LLM backend

- Ollama running on a rented GPU, reached via SSH tunnel. **Step 0: verify the endpoint before writing anything else.** Write `attack/check_endpoint.py` that probes `http://localhost:8080/v1/models`, `http://localhost:8080/api/tags`, `http://localhost:11434/v1/models`, `http://localhost:11434/api/tags` and reports which one is a working Ollama/OpenAI-compatible API. If 8080 returns HTML (Open WebUI), stop and tell me to forward port 11434 instead.
- Use the **OpenAI-compatible** `/v1/chat/completions` API. Base URL and model name configurable via CLI flags and env vars (`ATTACK_BASE_URL`, `ATTACK_MODEL`). Default model: `qwen2.5:32b` (check `/api/tags` for what's actually pulled).
- Structured output: pass a JSON schema via `response_format` (`{"type":"json_schema", ...}`). If the server rejects it, fall back to Ollama's native `/api/chat` with `format=<schema>`.
- `temperature=0`, `seed=42`.
- **Disk cache**: key = sha256(model + prompt version + full prompt text); store responses as JSON in `attack/cache/`. Reruns must not call the LLM again.
- Concurrency: thread pool, 8 workers (configurable). Retry 3× with exponential backoff on network errors or invalid JSON, then log and skip the record — never crash the whole run.

## Module layout

```
attack/
  run.py            CLI entry point
  check_endpoint.py
  llm.py            client, cache, retries, concurrency
  extract.py        pydantic schema + extraction prompt + few-shots
  verify.py         evidence verification
  tables.py         frequency lookup tables + normalization
  score.py          bits, λ, risk, aggregation
  recover.py        recovery attack (needs --original)
  report.py         summary.json, report.md, bar chart
  tests/            pytest
  README.md, requirements.txt
```

CLI:
```
python -m attack.run --input <anon.jsonl> [--original <orig.jsonl>] [--model ...] [--base-url ...]
                     [--out attack/results/<run_name>] [--limit N] [--workers 8]
```
`--limit` processes only the first N records (for iteration).

## Step 1 — Loader
Join `dialogue` + `note` into one text (the attacker sees both). Keep `encounter_id`. Count placeholders per record with the regex → `masking_density` (placeholders per 1k chars). Pass ICD codes along but they are scored separately (see scoring).

## Step 2 — Extraction (one LLM call per record)

Pydantic schema:
```python
class Attr(BaseModel):
    value: str | None
    evidence: str | None   # verbatim quote from the text
    rationale: str | None  # one line: why this narrows down who the patient is

class OtherFact(Attr):
    rarity: Literal["common", "uncommon", "rare", "very_rare"]

class Extraction(BaseModel):
    age: Attr
    birth_year: Attr
    sex: Attr
    state_or_region: Attr
    city: Attr
    occupation: Attr
    employer_or_facility: Attr
    ethnicity: Attr
    nationality: Attr
    absolute_dates: list[Attr]     # any date with a year or a month
    family_structure: Attr         # e.g. "10 siblings", "5 children"
    other_facts: list[OtherFact]   # idiosyncratic facts: hobbies, owns a business, veteran, rare events...
```

Prompt rules (put them in a system prompt, version it with a `PROMPT_VERSION` constant):
- Extract only facts about **the patient**, not the clinician or other people.
- Placeholders in square brackets are masked values: never output a placeholder as a value; treat the field as unknown.
- Generalized values (`[40-50]`, `90+`, `[DOB] 1982`) → extract what is actually stated (a range, a year).
- If a field is not stated, use `null`. Do **not** guess here (guessing belongs to the recovery attack).
- Every non-null value must have a **verbatim** evidence quote copied from the text.
- `rarity` meaning: common ≈ 1 in 10 people, uncommon ≈ 1 in 100, rare ≈ 1 in 1,000, very_rare ≈ 1 in 10,000 or rarer.
- Do not put diagnoses, symptoms, medications, or lab results in `other_facts`.
- Add 2–3 **short** few-shot examples written from this dataset's style, including one where almost everything is `null`.

## Step 3 — Evidence verification
Normalize text and quote (lowercase, collapse whitespace, strip punctuation at ends). Accept if the quote is a substring; otherwise accept if `rapidfuzz.fuzz.partial_ratio >= 90`. Drop failing attributes and count them → `hallucination_rate` (dropped / total extracted).

## Step 4 — Normalization and lookup tables (`tables.py`)
Hard-coded approximate US frequencies, with a comment citing the source (US Census 2020/ACS, BLS OES). Approximate values are fine; document that. Every scored attribute gets `p` and `p_source` ∈ {`census`, `bls`, `bucket`, `constant`}.

| Category | p |
|---|---|
| exact age | single-year share from a coarse census age table (~0.012–0.015) |
| age range / `90+` | sum of single-year shares in the range |
| birth_year + an absolute event year | derive exact age, count **once** under age |
| sex | 0.5 |
| state | census population share (all 50 states + DC) |
| region / country | region ≈ 0.2, country ≈ 1 |
| city | 1e-3 (constant) |
| occupation | ~60 common occupations with BLS employment share; alias map (e.g. "x-ray tech" → radiologic technologist); fallback to SOC major-group share; unknown → 1e-2 |
| employer_or_facility (named) | 1e-4 (constant) |
| ethnicity | census shares |
| nationality | constant 0.1 unless American (→ 1) |
| absolute date | full date 1/365; month+year 1/12; year only → 1 (no info alone) |
| family_structure | bucket via simple rules, default 1e-2 |
| other_facts | common 1e-1, uncommon 1e-2, rare 1e-3, very_rare 1e-4 |

Anti double-counting:
- One value per category; keep the most specific (exact age beats a range).
- Keep at most the **3 rarest** `other_facts`.
- Only the single most specific absolute date counts.

## Step 5 — Scoring (`score.py`)
Per record, over verified attributes:
```
bits   = Σ −log2(p_i)
λ      = max((N − 1) · Π p_i, 1e-12)      N = 335e6
risk   = (1 − exp(−λ)) / λ                 # P(attacker picks the right person); Poisson assumption
```
Diagnoses (ICD codes) are **not** part of risk: they must be preserved by the task rules, so they are a floor no pipeline can lower. Just report `n_icd_codes` per record.

Dataset summary (`summary.json`):
- `mean_risk` (headline, lower = better), `median_bits`, `p90_bits`
- `pct_likely_unique` (share of records with λ < 1)
- `top_leaking_attributes`: total bits per category across the dataset, sorted
- `hallucination_rate`, `mean_masking_density`, `n_records`, `n_failed`
- with `--original`: the same metrics on the original file, plus `privacy_gain = 1 − mean_risk_anon / mean_risk_orig` and `bits_removed = mean_bits_orig − mean_bits_anon`

## Step 6 — Recovery attack (`recover.py`, only with `--original`)
- Second prompt on the anonymized text: "This text was anonymized. For each masked placeholder about the patient, give your best guess of the original value and your reasoning." Structured output: list of `{placeholder, guess, reasoning, confidence}` restricted to types: name, age, state_or_region, city, year, employer_or_facility, occupation.
- Ground truth: run the Step 2 extraction on the **original** record (reuse the code and cache).
- Matching: age ±1; state/year exact (normalize case, abbreviations); names/employer/occupation via `rapidfuzz.fuzz.token_set_ratio >= 85`.
- Output `recovery_rate` per type and overall, plus the 5 best successful recoveries with the model's reasoning.

## Step 7 — Outputs and explainability
- `per_record.jsonl`: encounter_id, every attribute with value/evidence/rationale/p/p_source/bits, dropped attributes, bits, λ, risk, masking_density, recovery results if any.
- `summary.json` as above.
- `report.md`: summary table (anon vs original when available), the 5 riskiest records with each identifying attribute explained, recovery examples, and `top_leaking_attributes.png` (matplotlib bar chart).

## Step 8 — Tests and validation
- pytest: scoring on a hand-made record with known p values (check bits, λ, risk, edge cases λ→0 and large λ); verification accepts exact/near quotes and rejects invented ones; placeholder regex matches all placeholder types listed above.
- Sanity check (print a warning if violated): risk on original > risk on anonymized.
- Iterate on `--limit 5` first, inspect `per_record.jsonl`, then run the full set.
- Expected findings to sanity-check extraction against the current anonymized eval output:
  - R7: an occupation term and an employer name survive.
  - R9: owner of a specific small business.
  - R8: veteran with a stated disability rating.
  - R10: unusually large number of siblings.
  - R1 (recovery): a named landmark next to `[US_STATE]` → the state.
  - R6 (recovery): `[DOB] YYYY` + stated age → visit year.

## README (`attack/README.md`)
Usage, outputs, metric definitions with formulas, and limitations:
- Patients aren't real: this measures expected uniqueness under population statistics, not actual re-identification.
- Independence assumption overestimates risk; partially mitigated by the de-duplication rules.
- Employer constant and rarity buckets are heuristics.
- Results depend on attacker strength; report model name with every result.
- The attack ran on a rented GPU because the dataset is public; real patient data would require an on-prem attacker.

## Working style
- Work in small steps; after Step 0 and after the first `--limit 5` run, show me the output before continuing.
- Keep dependencies minimal: `requests` or `openai`, `pydantic`, `rapidfuzz`, `matplotlib`, `pytest`.
