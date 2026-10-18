# Re-identification attack

Bonus task "Adversarial Privacy Attacks". This is an attacker that reads the anonymized JSONL,
estimates how identifiable each patient still is, and, when the original file is available, tries to
recover what the placeholders hid. It is fully independent of the anonymization pipeline: it does not
import it, and it has its own dependencies.

## Status

Implemented and validated end to end on five real records; see [REPORT.md](REPORT.md) for results,
timings, the bugs found during validation and what is left. The full-dataset runs were not completed.

## Setup: run on the GPU host (recommended)

The repository is synced to the GPU host and the attack runs there, next to Ollama. No API is exposed;
only SSH is used (the host's SSH port is non-standard):

```bash
rsync -az -e "ssh -p 12621" --exclude .venv --exclude attack/cache --exclude attack/results \
      ./ root@<gpu-host>:/workspace/progetto-hackaton/      # or: git clone on the host
ssh -p 12621 root@<gpu-host>
cd /workspace/progetto-hackaton
uv venv /workspace/venv && . /workspace/venv/bin/activate
uv pip install -r attack/requirements.txt         # the attack alone
python -m attack.check_endpoint --chat-test       # finds the API, lists pulled models, one test reply
python -m attack.run --original --limit 5 --workers 4
```

Copy the results back with
`rsync -az -e "ssh -p 12621" root@<gpu-host>:/workspace/progetto-hackaton/attack/results/ attack/results/`.

To attack a fresh pipeline output, anonymize first, then point the attack at it (the attack never
imports the pipeline; it only reads its JSONL). The pipeline needs its own requirements
(`uv pip install --index-strategy unsafe-best-match -r requirements.txt`):

```bash
medanon anonymize -i data/<file>.jsonl -o outputs/<file>.anonymized.jsonl
python -m attack.run --input outputs/<file>.anonymized.jsonl --original data/<file>.jsonl --workers 4
```

Do not run both at once: the pipeline is CPU-only but saturates the cores, and Ollama's
schema-constrained decoding needs CPU too (load ~120 on 16 cores left the GPU at 0–11 %). Set
`OLLAMA_NUM_PARALLEL` on the host to at least `--workers`, or the extra workers only queue.

### Alternative: local run through an SSH tunnel

```bash
python -m venv .venv && .venv/bin/pip install -r attack/requirements.txt
ssh -N -p <port> -L 11434:localhost:11434 <user>@<gpu-host>
python -m attack.check_endpoint --chat-test
```

`check_endpoint` probes ports 8080 and 11434. If 8080 answers with HTML, that is Open WebUI, not the
API: forward 11434 instead.

## Usage

```bash
python -m attack.run --input data/output_eval_patient_doctor_encounters.jsonl \
                     --original data/eval_patient_doctor_encounters.jsonl --limit 5
python -m attack.run --original            # bare flag = default original path; full dataset
```

| flag | default |
|---|---|
| `--input` | `data/output_eval_patient_doctor_encounters.jsonl` |
| `--original` | off (enables the comparison and the recovery attack) |
| `--model` / `ATTACK_MODEL` | `qwen3.5:35b` (the model pulled on the GPU host; the spec suggested `qwen2.5:32b`) |
| `--base-url` / `ATTACK_BASE_URL` | `http://localhost:11434/v1` |
| `--backend` | `auto`: Ollama → native `/api/chat`, anything else → `/v1/chat/completions` |
| `--num-ctx` | 16384 (Ollama context window) |
| `--out` | `attack/results/<model>_<input>[_limitN]` |
| `--limit`, `--workers` | all records, 8 threads |

The LLM runs with temperature 0, seed 42 and a JSON schema for the output (`response_format`, or Ollama
`format`). Every response is cached in `attack/cache/`, keyed by sha256(model + prompt version + prompt),
so reruns make no LLM calls. Failed calls are retried 3 times with exponential backoff. The seed shifts on
each retry, because at temperature 0 the same seed would replay the same invalid output. After the last
retry the record is logged as failed and the run continues.

**Why native Ollama by default:** Ollama's OpenAI-compatible endpoint cannot set the context size. Prompts
longer than the server default (2–4k tokens) are truncated silently, and our prompts reach about 9k tokens.
The native API sets `num_ctx`. Use `--backend openai` to force `/v1/chat/completions`. It still falls back
to `/api/chat` if the server rejects `response_format`.

**Thinking models** (e.g. `qwen3.5`): the client reads the model's capabilities from `/api/show` and sends
`think: false`. With thinking on, one extraction took 280 s and 5.5k tokens instead of 37 s and 700 tokens,
for the same JSON.

**Timing** (`qwen3.5:35b` Q4 on one RTX 4090, host otherwise idle): extraction ≈ 3.7 s + recovery ≈ 3.4 s,
so ≈ 7–8 s to attack one anonymized record; ≈ 11–12 s with `--original`. With MedAnon running on the same
host, the same calls took 33 s + 9 s, because schema-constrained decoding needs CPU. Calls are cached, so
a rerun only pays for prompts that changed.

## Pipeline

1. **Load.** The attacker sees `dialogue + note`. Placeholders (`\[[A-Z][A-Z_]*(?:_\d+)?(?::[^\]]*)?\]`)
   are counted to get `masking_density` = placeholders per 1,000 characters.
2. **Extract.** One LLM call per record fills a fixed schema with values about the patient only: age,
   birth year, sex, state/region, city, occupation, employer/facility, ethnicity, nationality, absolute
   dates, family structure, and other idiosyncratic facts with a rarity bucket. Every value carries a
   verbatim evidence quote and a one-line rationale. Placeholders count as unknown, and the extractor is
   told not to guess.
3. **Verify.** The evidence must be a substring of the normalized text, or match it with
   `rapidfuzz.partial_ratio ≥ 90`. Attributes that fail are dropped. For fields that should be literal
   (state, city, employer, birth year, dates), the value must also appear in its own evidence. This catches
   inferred guesses such as a US state "quoted" from a landmark name. Two guards enforce prompt rules
   that models still break: `other_facts` that are medical (keyword list, or a fuzzy match with the
   record's ICD descriptions) are dropped as `medical_fact`; person names (a first and last name, "full name …") as `name_not_scored`; relative times given as absolute dates ("last weekend", "4 days
   ago") as `not_absolute_date`; facts that restate an external-cause or activity ICD code (golf next to
   `Y93.53 Activity, golf`, "rear-ended" next to a motor-vehicle accident code) as `icd_floor`. None of
   these counts as a hallucination. A quote that joins fragments with "..." is checked fragment by
   fragment.
4. **Price.** Every kept attribute gets a probability `p`, with `p_source` ∈ {census, bls, bucket,
   constant} (see `tables.py`):

   | attribute | p |
   |---|---|
   | exact age / range / `90+` | single-year shares from ACS 2020 age groups, summed over the range |
   | birth year | one-year birth cohort; with an absolute event year, the exact age. Counted once, as age |
   | sex | 0.5 |
   | state / region / country | Census 2020 share / 0.2 / US = 1 |
   | city | 1e-3 |
   | occupation | BLS OEWS May 2023 employment share, with aliases ("x-ray tech" → radiologic technologist); otherwise the SOC major-group share; unknown → 1e-2; owner of a specific business → 1e-3 |
   | employer or facility | named 1e-4, generic ("the local hospital") 0.1 |
   | ethnicity | Census 2020 shares (unknown 0.05) |
   | nationality | 0.1, American = 1 |
   | absolute date | day 1/365, month 1/12, year only 1 |
   | family structure | buckets ("ten siblings" → 1e-3); default 1e-2 |
   | other facts | common 1e-1, uncommon 1e-2, rare 1e-3, very rare 1e-4; a fact naming a US state ("hikes in <state>") is priced as that state |

   Anti double-counting: one value per category (the most specific, i.e. lowest p); only the most
   specific absolute date; at most the 3 rarest other facts, where a fact restating a counted one
   (overlapping evidence and a shared word stem: "hiking" / "hikes regularly") is skipped as
   `duplicate_of`.
5. **Score**, per record, over the counted attributes, with N = 335e6:

   ```
   bits = Σ −log2(p_i)
   λ    = max((N − 1) · Π p_i, 1e-12)     expected number of other people sharing every attribute
   risk = (1 − e^−λ) / λ                  P(attacker picks the right person), Poisson look-alikes
   ```

   Diagnoses are not part of the risk: the task requires keeping them, so they set a floor that no
   pipeline can lower. `n_icd_codes` is reported per record instead.
6. **Recover** (needs `--original`). A second prompt asks for a best guess for each placeholder about the
   patient (name, age, state/region, city, year, employer/facility, occupation), with reasoning and
   confidence. To build the ground truth, the anonymized and original texts are aligned word by word, which
   gives the exact text each placeholder replaced. When a guessed placeholder cannot be aligned, the guess
   is compared with the Step 2 extraction of the original instead. Matching rules: age ±1; states and years
   exact, after normalizing case and abbreviations; other regions and landmarks ("North Cascades" vs
   "cascades"), names, employer, occupation and city with `token_set_ratio ≥ 85`, after removing titles
   (and generic words such as "hospital" for employers). The summary reports the rate overall and without
   names, which rarely have any clue in the text. A
   guess whose truth cannot be compared (a year guess for a masked "next week") is *unscorable*. It is left
   out of the rate but listed in the report. So are guesses for placeholders that do not occur in the text
   (`placeholder_not_in_text`) and for labels outside the attack's types (`out_of_scope`: weekday, month,
   holiday, ID, ethnicity, nationality, clinician). Filler names (generic placeholder names, the dictation software's name)
   are scored as wrong, never as a lucky match on a common first name.

## Outputs (`attack/results/<run>/`)

- `per_record.jsonl`: for each attribute, the value, evidence, rationale, p, p_source, bits and whether it
  is counted. Also the dropped attributes with a reason, bits, λ, risk, masking_density, n_icd_codes and
  the recovery guesses with their truth. `per_record_original.jsonl` holds the same for the original file.
- `summary.json` contains:
  - `mean_risk` (headline, lower = better), `median_bits`, `p90_bits`, `mean_bits`
  - `pct_likely_unique` (% of records with λ < 1)
  - `top_leaking_attributes` (total bits per category)
  - `hallucination_rate` (dropped for evidence / total extracted), `mean_masking_density`, `n_records`,
    `n_failed`
  - with `--original`: the same metrics for the original, plus `privacy_gain = 1 − risk_anon/risk_orig`,
    `bits_removed = bits_orig − bits_anon` (both computed on records where both extractions succeeded),
    `recovery` (rates overall, without names and per type; best recoveries) and sanity `warnings` (e.g. a record riskier
    anonymized than original)
- `report.md` and `top_leaking_attributes.png`: summary table, the 5 riskiest records with every
  identifying attribute explained, recovery examples.

**How the comparison is scored (`--original`).** Two LLM extractions of near-identical texts disagree:
on the validation records the anonymized extraction missed facts ("commutes by bus") that were still in
the anonymized text, and the original's missed facts that were in the original. Compared naively, that
noise looks like privacy gain (or loss). So each side is scored on its own extraction plus the other
side's facts whose evidence verifies in its own text: anonymized-side facts are checked against the
original (`from_anonymized`), original-side facts against the anonymized text, verbatim only
(`from_original`), so a quote that contains a masked value ("went to <state>") can never match
"went to [US_STATE]". `bits_removed` then counts only what the anonymizer removed. Without `--original`,
the anonymized record is scored on its own extraction. Hallucination and failure counts always use
each file's own extraction.

## Deterministic tools (no LLM)

- `python -m attack.audit_masks --original orig.jsonl --anonymized anon.jsonl --out audit.json [--examples ex.jsonl]`
  aligns each original with its anonymized version and counts what every placeholder replaced
  (identifier, name, date, clinical term, measurement), plus residue in the anonymized text: exact adult
  ages, bare answers to "how old are you?", 4/2-digit birth years, and the age information in bits.
  `audit.json` holds counts only; `--examples` contains original text, for local review only.
- `python -m attack.export_predictions <run>/per_record.jsonl -o predictions.jsonl` writes attacker-side
  predictions only (no ground truth, no facts borrowed from the original, risk re-scored), for
  comparison with another attacker on the same anonymized file.

## Tests and lint

`ruff check attack` (config in `attack/ruff.toml`; ruff is a dev tool, not a requirement).
`python -m pytest attack/tests` covers scoring (known p, λ → 0, large λ, de-duplication), evidence
verification, the placeholder regex, the lookup tables, alignment and matching, and an offline end-to-end
run with a fake LLM.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `no LLM API at …` | the tunnel or server is down: run `python -m attack.check_endpoint`; on the host, `curl localhost:11434/api/tags` |
| `model '…' is not available; pulled: …` | pass one of the listed models with `--model`, or `ollama pull` it |
| HTML from port 8080 | that is Open WebUI: forward or use 11434 |
| `prompt + output reached num_ctx` | raise `--num-ctx`; Ollama would otherwise shift the context and drop the start of the prompt |
| very slow calls | thinking left on (old Ollama ignoring `think`), `OLLAMA_NUM_PARALLEL=1` with several workers, or CPU contention: schema-constrained sampling runs on the CPU, so don't run the pipeline at the same time |
| records listed as failed | invalid JSON after 4 attempts (logged); rerun: successful calls come from the cache |
| results did not change after a prompt edit | bump `PROMPT_VERSION` in `extract.py` / `recover.py`; the cache key includes it |
| `SANITY:` warnings | a record scores riskier anonymized than original; inspect both `per_record*.jsonl` rows |

## Limitations

- The patients are not real, so this measures expected uniqueness under population statistics, not actual
  re-identification.
- Attributes are assumed independent, which overestimates risk (age and occupation correlate, for
  example). The de-duplication rules only partly mitigate this.
- The employer constant, the rarity buckets, the family buckets and the city/region constants are
  heuristics. The tables are rounded on purpose.
- Direct identifiers (names, IDs, addresses) are not in the extraction schema, so the original's risk is a
  lower bound and `privacy_gain` understates what masking names achieves. Names are evaluated by the
  recovery attack instead.
- Results depend on the attacker's strength: always report the model name (it is stored in
  `summary.json` and in the report).
- The attack ran on a rented GPU because the dataset is public. Real patient data would require an
  on-prem attacker.
