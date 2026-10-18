"""Attacker-side predictions only, for comparison with an independent attacker (e.g. a Splink linkage
benchmark) on the same anonymized JSONL. Drops everything that came from the original: facts borrowed
by the comparison union (`from_original`), recovery ground truth and match verdicts.

    python -m attack.export_predictions attack/results/<run>/per_record.jsonl -o predictions.jsonl

One line per encounter:
{"encounter_id", "risk", "lambda", "bits",
 "quasi_identifiers": [{"category", "value", "evidence", "p", "p_source", "bits", "counted"}],
 "recovery_guesses": [{"placeholder", "type", "guess", "confidence"}]}
"""
import argparse
import json

from .score import score_record

QI_KEYS = ("category", "value", "evidence", "p", "p_source", "bits", "counted")
GUESS_KEYS = ("placeholder", "type", "guess", "confidence")


_DERIVED = {"category", "p", "p_source", "note", "bits", "counted", "duplicate_of"}


def export(row: dict) -> dict:
    """Risk is re-scored from the attacker's own facts, so it never includes borrowed ones."""
    if row.get("failed"):
        return {"encounter_id": row["encounter_id"], "failed": True, "quasi_identifiers": [], "recovery_guesses": []}
    own = [{k: v for k, v in a.items() if k not in _DERIVED} for a in row.get("attributes", [])
           if not a.get("from_original")]
    scored = score_record(own)
    return {"encounter_id": row["encounter_id"], "risk": scored["risk"], "lambda": scored["lambda"],
            "bits": scored["bits"], "failed": False,
            "quasi_identifiers": [{k: a.get(k) for k in QI_KEYS} for a in scored["attributes"]],
            "recovery_guesses": [{k: g.get(k) for k in GUESS_KEYS} for g in row.get("recovery") or []]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m attack.export_predictions", description=__doc__.splitlines()[0])
    ap.add_argument("per_record")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args(argv)
    with open(args.per_record, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            out.write(json.dumps(export(json.loads(line)), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
