"""Checks an anonymized output of encounters.jsonl against expectations.json.

    python check_regressions.py <anonymized.jsonl> [--json result.json]

Exit code 0 only if every clinical assertion passes (non-regression gate). Privacy assertions are
reported separately: they document known weaknesses, so some are expected to fail until fixed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent


def check(anonymized: str) -> list[dict]:
    recs = {}
    for line in open(anonymized, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["encounter_id"]] = (r.get("dialogue") or "") + "\n" + (r.get("note") or "")
    results = []
    for e in json.loads((HERE / "expectations.json").read_text(encoding="utf-8")):
        text = recs.get(e["encounter_id"])
        found = text is not None and re.search(e["regex"], text, re.I) is not None
        ok = text is not None and found == (e["expect"] == "present")
        results.append({**e, "pass": ok, "missing_record": text is None})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anonymized")
    ap.add_argument("--json")
    args = ap.parse_args()
    results = check(args.anonymized)
    for kind in ("privacy", "clinical"):
        rs = [r for r in results if r["kind"] == kind]
        print(f"{kind}: {sum(r['pass'] for r in rs)}/{len(rs)} pass")
        for r in rs:
            if not r["pass"]:
                print(f"  FAIL {r['encounter_id']:16} {r['finding']:34} expect {r['expect']:7} /{r['regex']}/")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["pass"] for r in results if r["kind"] == "clinical") else 1


if __name__ == "__main__":
    sys.exit(main())
