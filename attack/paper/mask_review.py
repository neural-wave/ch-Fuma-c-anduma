"""Human review of MedAnon false-positive masks on the 67 evaluation encounters.

Every placeholder whose aligned original span is not a person name, date or other identifier was read in
context by a reviewer (one annotator) and labelled below. Only non-identifying spans are stored, so this
file holds no patient data. Labels:
  clinical_fp  clinical content masked (exam finding, anatomy, eponym/test, drug, device, study, diagnosis)
  filler_fp    filler word or transcription artifact masked ("lem" = "let me", "mm hmm", "hey dragon")
  other_fp     non-identifying, non-clinical word masked ("cold turkey", "french fry")
Spans not listed here were judged correct masks (names, places, organizations, dates, occupations, IDs).

    .venv/bin/python attack/paper/mask_review.py --original data/eval_patient_doctor_encounters.jsonl \
        --anonymized outputs/eval_paper/eval.age_bands.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from attack.audit_masks import replacements

CLINICAL = {  # (encounter_id, aligned original span, lowercase) -> what was lost
    ("D2N003", "lungs"): "exam: lungs", ("D2N016", "lungs"): "exam: lungs", ("D2N058", "* your lungs"): "exam: lungs",  # span starts with a name: matched by suffix
    ("D2N006", "autoimmune labs"): "study: autoimmune labs", ("D2N035", "labs"): "study: labs",
    ("D2N014", "supple"): "exam: supple neck", ("D2N027", "pursed"): "exam: pursed-lip breathing",
    ("D2N041", "coarse"): "exam: coarse rhonchi", ("D2N028", "normoactive"): "exam: normoactive bowel sounds",
    ("D2N059", "integumentary"): "exam: integumentary system", ("D2N060", "afebrile"): "exam: afebrile",
    ("D2N048", "neurovascular"): "exam: neurovascular status",
    ("D2N022", "fds"): "anatomy: flexor digitorum superficialis", ("D2N022", "fdp"): "anatomy: flexor digitorum profundus",
    ("D2N058", "rca"): "anatomy: right coronary artery (stent)",
    ("D2N044", "lachman"): "test: Lachman", ("D2N044", "achilles"): "exam: Achilles reflex",
    ("D2N029", "clark"): "staging: Clark level", ("D2N029", "breslow"): "staging: Breslow thickness",
    ("D2N029", "shave"): "procedure: shave biopsy", ("D2N032", "mr arthrogram"): "study: MR arthrogram",
    ("D2N030", "buspar"): "drug: Buspar", ("D2N030", "camila"): "drug: Camila (norethindrone)",
    ("D2N046", "taxol"): "drug: Taxol", ("D2N042", "aircast"): "device: Aircast", ("D2N065", "aircast"): "device: Aircast",
    ("D2N022", "band aid"): "supply: band-aid", ("D2N022", "occupational therapist"): "care: occupational therapy",
    ("D2N025", "preop"): "care: pre-operative examination",
    ("D2N021", "cushing's"): "diagnosis: Cushing's", ("D2N038", "hep c"): "diagnosis: hepatitis C",
    ("D2N055", "toret"): "diagnosis: Tourette's", ("D2N056", "farmer's"): "diagnosis: farmer's lung",
}
FILLER = {("D2N039", "lem"), ("D2N046", "lem"), ("D2N053", "lem"), ("D2N059", "lem"), ("D2N060", "lem"),
          ("D2N018", "mm hmm"), ("D2N026", "mm hmm"), ("D2N020", "hey"), ("D2N020", "hey hey"),
          ("D2N043", "bye bye"), ("D2N064", "bye bye"), ("D2N054", "lucken"), ("D2N027", "exona")}
OTHER = {("D2N043", "turkey"): "idiom: cold turkey", ("D2N049", "french"): "food: french fry",
         ("D2N062", "carbo"): "transcription: cargo", ("D2N057", "mexican's"): "food: Mexican food"}


def label_of(eid: str, span: str) -> tuple[str, str] | None:
    s = re.sub(r"\s+", " ", span.lower()).strip()
    for (e, key), what in CLINICAL.items():  # "* suffix" keys avoid storing a name that precedes the term
        if e == eid and key.startswith("* ") and s.endswith(key[2:]):
            return "clinical_fp", what
    if (eid, s) in CLINICAL:
        return "clinical_fp", CLINICAL[(eid, s)]
    if (eid, s) in FILLER:
        return "filler_fp", s
    if (eid, s) in OTHER:
        return "other_fp", OTHER[(eid, s)]
    return None


def review(original: str, anonymized: str) -> dict:
    orig = {json.loads(line)["encounter_id"]: json.loads(line) for line in open(original)}
    blocks = Counter()
    records = defaultdict(set)
    lost = Counter()
    labels_seen = Counter()
    for line in open(anonymized):
        r = json.loads(line)
        o = orig[r["encounter_id"]]
        for field in ("dialogue", "note"):
            for x in replacements(r[field], o[field]):
                lab = label_of(r["encounter_id"], x["original_span"])
                kind = lab[0] if lab else "correct_or_identifier"
                blocks[kind] += 1
                if lab:
                    records[kind].add(r["encounter_id"])
                    labels_seen[(r["encounter_id"], x["original_span"].lower())] += 1
                    if kind == "clinical_fp":
                        lost[lab[1].split(":")[0]] += 1
    expected = set(CLINICAL) | FILLER | set(OTHER)
    missing = sorted(k for k in expected if not any(
        k[0] == e and (k[1] == s or (k[1].startswith("* ") and s.endswith(k[1][2:]))) for e, s in labels_seen))
    n = len(orig)
    return {"n_records": n, "placeholder_blocks": dict(blocks),
            "records_with": {k: f"{len(v)}/{n}" for k, v in records.items()},
            "record_ids": {k: sorted(v) for k, v in records.items()},
            "clinical_fp_blocks_by_kind": dict(lost.most_common()),
            "clinical_fp_terms": sorted({v for v in CLINICAL.values()}),
            "labels_not_found_in_this_output": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--anonymized", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = review(args.original, args.anonymized)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("record_ids", "clinical_fp_terms")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
