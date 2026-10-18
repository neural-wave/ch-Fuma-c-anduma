"""Local human-annotation workflow for the fixed validation subset (real dev encounters).

    python scripts/annotation.py export            # -> outputs/annotation/tasks.json (Label Studio import)
    python scripts/annotation.py score --annotations outputs/annotation/export.json

1. `export` writes one Label Studio task per encounter ("DIALOGUE:" + dialogue + "NOTE:" + note).
   Default is BLIND (no pre-annotations) so annotators are not anchored to any detector;
   `--preannotate` adds the baseline's spans as editable predictions (faster, but biased).
2. Annotate locally (pip install label-studio; label-studio start - runs offline on localhost) with
   the labels below, mark every span that is privacy-relevant. Never upload the data anywhere.
3. `score` maps annotated spans back to (field, offset) and scores every configuration whose spans
   are in outputs/experiments/<config>.spans.json: precision / recall / F1 with Wilson CIs, by label
   and field. Files stay under outputs/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LABELS = ["PERSON", "LOCATION", "ORGANIZATION", "OCCUPATION", "AGE", "DATE", "DOB", "ID", "CONTACT", "ADDRESS", "OTHER_PHI"]
SEP = "\n\nNOTE:\n"
LABEL_CONFIG = ('<View><Labels name="phi" toName="text">'
                + "".join(f'<Label value="{label}"/>' for label in LABELS)
                + '</Labels><Text name="text" value="$text"/></View>')


def _records():
    ids = set(json.loads((ROOT / "configs/validation_subset.json").read_text())["encounter_ids"])
    with open(ROOT / "data/patient_doctor_encounters.jsonl", encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r["encounter_id"] in ids]


def export(preannotate: bool) -> None:
    out = ROOT / "outputs/annotation"
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "outputs/experiments/baseline.spans.json").read_text()) if preannotate else {}
    tasks = []
    for r in _records():
        head = "DIALOGUE:\n"
        text = head + r["dialogue"] + SEP + r["note"]
        task = {"data": {"text": text, "encounter_id": r["encounter_id"]}}
        if preannotate:
            res = []
            for fld, s, e, label, action, _ in base.get(r["encounter_id"], []):
                if action == "keep" or label not in LABELS:
                    continue
                off = len(head) if fld == "dialogue" else len(head) + len(r["dialogue"]) + len(SEP)
                res.append({"from_name": "phi", "to_name": "text", "type": "labels",
                            "value": {"start": off + s, "end": off + e, "labels": [label]}})
            task["predictions"] = [{"model_version": "medanon-baseline", "result": res}]
        tasks.append(task)
    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    (out / "label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    print(f"wrote {len(tasks)} tasks to {out / 'tasks.json'} (blind={not preannotate}) and label_config.xml")


def _gold(export_path: str) -> dict:
    recs = {r["encounter_id"]: r for r in _records()}
    gold = {}
    for task in json.loads(Path(export_path).read_text(encoding="utf-8")):
        eid = task["data"]["encounter_id"]
        r = recs[eid]
        d0 = len("DIALOGUE:\n")
        n0 = d0 + len(r["dialogue"]) + len(SEP)
        spans = []
        for ann in (task.get("annotations") or [])[:1]:
            for x in ann.get("result", []):
                v = x["value"]
                s, e, lab = v["start"], v["end"], v["labels"][0]
                if s >= n0:
                    spans.append(("note", s - n0, e - n0, lab))
                elif s >= d0:
                    spans.append(("dialogue", s - d0, e - d0, lab))
        gold[eid] = spans
    return gold


def score(export_path: str) -> None:
    from medanon.evaluation.scoring import wilson

    gold = _gold(export_path)
    for spans_file in sorted((ROOT / "outputs/experiments").glob("*.spans.json")):
        pred = json.loads(spans_file.read_text())
        tp = fn = fp = 0
        by = Counter()
        for eid, g in gold.items():
            p = [s for s in pred.get(eid, []) if s[4] != "keep" and s[3] != "LAB_VALUE"]
            for fld, s, e, lab in g:
                hit = any(x[0] == fld and x[1] < e and s < x[2] for x in p)
                tp += hit
                fn += not hit
                by[(lab, fld, hit)] += 1
            for x in p:
                fp += not any(fld == x[0] and s < x[2] and x[1] < e for fld, s, e, _ in g)
        rec = tp / max(1, tp + fn)
        n_pred = sum(len([s for s in pred.get(eid, []) if s[4] != "keep" and s[3] != "LAB_VALUE"]) for eid in gold)
        prec = (n_pred - fp) / max(1, n_pred)
        print(f"{spans_file.stem.replace('.spans', ''):<26} recall {rec:.3f} {wilson(tp, tp + fn)}  precision {prec:.3f} "
              f"{wilson(n_pred - fp, n_pred)}  gold={tp + fn} pred={n_pred}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--preannotate", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--annotations", required=True)
    args = ap.parse_args()
    export(args.preannotate) if args.cmd == "export" else score(args.annotations)


if __name__ == "__main__":
    main()
