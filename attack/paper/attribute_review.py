"""Human review of the attacker's counted attributes (precision of the extraction + verification stage).

One reviewer read every counted attribute other than age and sex, with its evidence, and labelled it:
  valid       a true, non-medical fact about the patient, counted once
  medical     clinical content (the task keeps it; not a quasi-identifier for this scoring)
  icd_floor   restates an ICD-10 external-cause/activity code (kept by the task rules)
  duplicate   restates another counted attribute
  trivial     non-informative or garbled ("working", "prefers fall weather")
  uncertain   may not be about the patient
Pricing problems on valid facts (off by roughly 2+ bits) are listed separately as `mispriced`.
Keys are (encounter_id, value prefix); values are attacker output on anonymized text, no names.

ROUND_1 = eval set (67), MedAnon ceff209 age_bands, attacker before the review-driven fixes (used to find
bugs, so it is NOT a held-out estimate). HELDOUT is filled from the held-out training sample.

    .venv/bin/python -m attack.paper.attribute_review <per_record.jsonl> --round round1|heldout [--out x.json]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROUND_1 = {
    "labels": {
        ("D2N001", "prefers fall weather"): "trivial", ("D2N005", "stopped at a four-way stop"): "icd_floor",
        ("D2N008", "working"): "trivial", ("D2N010", "sits in the back of a car"): "trivial",
        ("D2N013", "has epilepsy"): "medical", ("D2N013", "recently had many visitors"): "trivial",
        ("D2N015", "lives in a house with a driveway"): "trivial", ("D2N016", "sees a counselor"): "medical",
        ("D2N018", "has gallstones"): "medical", ("D2N018", "had lumbar fusion"): "medical",
        ("D2N020", "nauseous when sitting"): "medical", ("D2N021", "on vacation after"): "trivial",
        ("D2N021", "has sensitive skin"): "medical", ("D2N023", "uses a med calendar"): "medical",
        ("D2N024", "uses Mederma"): "medical", ("D2N025", "uses pickle juice"): "medical",
        ("D2N025", "regularly undergoes EKGs"): "medical", ("D2N026", "uses a wheelchair"): "medical",
        ("D2N028", "sister with IBS"): "medical", ("D2N030", "works supporting Covid"): "duplicate",
        ("D2N032", "has PCOS"): "medical", ("D2N035", "drives a car (implied"): "trivial",
        ("D2N039", "employed"): "trivial", ("D2N041", "likes peanut grooves"): "duplicate",
        ("D2N043", "started smoking in high"): "duplicate", ("D2N044", "has a parking lot"): "trivial",
        ("D2N051", "stopped smoking 4 years"): "duplicate", ("D2N053", "uses T-Gel"): "medical",
        ("D2N056", "does not wear a mask"): "duplicate", ("D2N059", "had tonsils removed"): "medical",
        ("D2N060", "plays in a local league"): "uncertain", ("D2N064", "enjoys taking lots of walks"): "duplicate",
    },
    "mispriced": {
        ("D2N002", "lives in a house with basement"): "basement ~40% of homes (~1.3 bits), priced 3.3",
        ("D2N007", "desk worker"): "office work ~12% (~3 bits), priced as unknown occupation 6.6",
        ("D2N008", "lives near mountains"): "vague, priced 6.6", ("D2N012", "has children"): "~60% of adults, priced 6.6",
        ("D2N017", "lives in a home without stairs"): "~50%, priced 6.6", ("D2N019", "recently moved"): "~10%/yr, priced 6.6",
        ("D2N022", "rear-ended in a car accident"): "priced as 'owns a car' 0.5", ("D2N026", "has a lady friend"): "priced 6.6",
        ("D2N030", "healthcare worker"): "~6% (~4 bits), priced 6.6", ("D2N031", "social drinker"): "~60%, priced 6.6",
        ("D2N033", "owns a trampoline"): "~10% of households, priced 6.6", ("D2N033", "has a kid"): "priced 6.6",
        ("D2N034", "renovating basement"): "priced 6.6", ("D2N035", "eats salt and vinegar"): "priced 6.6",
        ("D2N039", "watches football games"): "priced 3.3", ("D2N040", "watched the Masters"): "priced 6.6",
        ("D2N041", "has little kids"): "priced 6.6", ("D2N041", "does not like sports"): "priced 3.3",
        ("D2N044", "works from home"): "~15% (2.7 bits), priced 6.6", ("D2N044", "lives in a house with an upstairs"): "priced 6.6",
        ("D2N051", "lives in a house"): "~65%, priced 3.3", ("D2N053", "has a sister"): "~80%, priced 6.6",
        ("D2N055", "lives with father"): "priced 6.6", ("D2N060", "daughter plays soccer"): "priced 6.6",
        ("D2N063", "lives a sedentary lifestyle"): "~25%, priced 3.3", ("D2N064", "works from home"): "~15%, priced 6.6",
    },
}
ROUNDS = {"round1": ROUND_1}
HELDOUT_FILE = Path(__file__).with_name("attribute_review_heldout.json")


def load_round(name: str) -> dict:
    if name == "heldout":
        raw = json.loads(HELDOUT_FILE.read_text())
        return {"labels": {tuple(k.split("|", 1)): v for k, v in raw["labels"].items()},
                "mispriced": {tuple(k.split("|", 1)): v for k, v in raw["mispriced"].items()}}
    return ROUNDS[name]


def evaluate(per_record: str, review: dict) -> dict:
    rows = [json.loads(line) for line in open(per_record) if line.strip()]

    def match(table: dict, eid: str, value: str):
        return next((v for (e, prefix), v in table.items() if e == eid and value.startswith(prefix)), None)

    labels, bits, used = Counter(), Counter(), set()
    mispriced = 0
    for r in rows:
        for a in r.get("attributes", []):
            if not a["counted"] or a["category"] in ("age", "sex") or a["bits"] <= 0:
                continue
            lab = match(review["labels"], r["encounter_id"], a["value"]) or "valid"
            labels[lab] += 1
            bits[lab] += a["bits"]
            if lab == "valid" and match(review["mispriced"], r["encounter_id"], a["value"]):
                mispriced += 1
            used.add(r["encounter_id"])
    n = sum(labels.values())
    return {"n_attributes": n, "labels": dict(labels), "precision": labels["valid"] / n if n else None,
            "bits_by_label": {k: round(v, 1) for k, v in bits.items()},
            "share_of_bits_from_invalid": 1 - bits["valid"] / sum(bits.values()) if bits else None,
            "valid_but_mispriced": mispriced}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("per_record")
    ap.add_argument("--round", default="round1", choices=["round1", "heldout"])
    ap.add_argument("--out")
    args = ap.parse_args()
    res = evaluate(args.per_record, load_round(args.round))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
