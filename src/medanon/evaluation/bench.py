"""Synthetic gold benchmark runner (safe to print: contains no real patient data)."""
from __future__ import annotations

import json
import re
import time
from collections import Counter

from .synthetic import generate, score


def run_synthetic(anonymizer, n: int = 120, seed: int = 7, styles=("aci", "discharge", "cased"), show_errors: int = 0) -> dict:
    report = {}
    for style in styles:
        recs = generate(n if style == "aci" else max(20, n // 3), seed=seed, style=style)
        t0 = time.perf_counter()
        outs, decs = [], []
        for r in recs:
            o, d = anonymizer.anonymize({k: v for k, v in r.items() if not k.startswith("_")})
            outs.append(o)
            decs.append(d)
        elapsed = time.perf_counter() - t0
        res = score(recs, outs, decs)
        res["seconds_per_record"] = round(elapsed / len(recs), 3)
        report[style] = res
        if show_errors:
            misses, spurious, traps = Counter(), Counter(), Counter()
            for r, o, d in zip(recs, outs, decs):
                repl = {(x["field"], i) for x in d if x["action"] != "keep" and x["label"] != "LAB_VALUE"
                        for i in range(x["start"], x["end"])}
                for g in r["_gold"]:
                    text = r[g["field"]]
                    if not all((g["field"], i) in repl for i in range(g["start"], g["end"]) if text[i].isalnum()):
                        ctx = r[g["field"]][max(0, g["start"] - 30):g["end"] + 20].replace("\n", " ")
                        misses[f"{g['label']:<12} {g['text']!r:<26} | {ctx}"] += 1
                gold = {(g["field"], i) for g in r["_gold"] for i in range(g["start"], g["end"])}
                for x in d:
                    if x["action"] != "keep" and x["label"] != "LAB_VALUE" and not any(
                            (x["field"], i) in gold for i in range(x["start"], x["end"])):
                        spurious[f"{x['label']:<12} {r[x['field']][x['start']:x['end']]!r:<22} via {','.join(x['detectors'])}"] += 1
                for t in r["_traps"]:
                    if any(x["action"] != "keep" and x["field"] == t["field"] and x["start"] < t["end"] and t["start"] < x["end"]
                           for x in d):
                        traps[t["text"]] += 1
            print(f"\n=== {style}: top misses")
            for k, v in misses.most_common(show_errors):
                print(f"{v:3d}  {k}")
            print(f"=== {style}: top spurious replacements")
            for k, v in spurious.most_common(show_errors):
                print(f"{v:3d}  {k}")
            print(f"=== {style}: damaged medical traps: {dict(traps.most_common(show_errors))}")
    return report


if __name__ == "__main__":
    import argparse

    from ..pipeline import Anonymizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--errors", type=int, default=15)
    ap.add_argument("--config")
    args = ap.parse_args()
    print(json.dumps(run_synthetic(Anonymizer(config_path=args.config), n=args.n, show_errors=args.errors), indent=1))
