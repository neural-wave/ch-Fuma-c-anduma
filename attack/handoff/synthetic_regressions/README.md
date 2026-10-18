# Synthetic regression probes for MedAnon

Fully invented encounters (people, places, values), one weakness per encounter, plus clinical facts that
must survive. No real or Bosch-provided text.

| file | role |
|---|---|
| `make_fixtures.py` | source of truth: encounters and assertions (edit here, then rerun) |
| `encounters.jsonl` | MedAnon input (Bosch schema: `encounter_id`, `dataset`, `dialogue`, `note`, `icd10_codes`) |
| `expectations.json` | assertions: `kind` privacy or clinical, `regex` (case-insensitive, on dialogue + note), `expect` absent or present |
| `check_regressions.py` | checker: exit 0 only if every **clinical** assertion passes; privacy failures are listed |
| `outputs/<policy>.jsonl` | actual MedAnon `ceff209` output (full detectors, `--offline`) for default, age_bands, age_bands_strict_dates |
| `outputs/<policy>.check.json` | per-assertion results for that output |

```bash
medanon anonymize -i encounters.jsonl -o out.jsonl --config configs/age_bands.yaml --offline
python check_regressions.py out.jsonl            # clinical must be all-pass; privacy shows what is still open
```

Gate for accepting a MedAnon change: clinical assertions stay 35/35+ (none may regress), and privacy
pass count does not decrease. A fix is confirmed when its probe's privacy assertions turn to pass.
Probe IDs map to findings in `../VULNERABILITIES.json` (`V*` measured, `L*` leads from the other attacker,
`C*` clinical counterexamples).
