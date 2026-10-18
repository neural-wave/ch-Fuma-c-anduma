"""Adversarial privacy attacks (bonus task). Each attack has an explicit threat model and is run
against the *original* release and the *anonymised* release; only aggregate rates are reported.

1. Linkage attack - the adversary knows a target's exact age, sex, occupation and home city
   (e.g. from social media) and searches the released corpus for consistent records.
   Metric: expected re-identification probability (mean 1/|candidates|) and share of targets
   that are uniquely re-identified.
2. Lab-value linkage - the adversary holds exact lab results (e.g. a leaked lab database) and
   links them to released encounters by exact (lab, value) matches.
3. Residual-name recovery - the adversary harvests every name-like token left in the release
   (capitalised or in a names lexicon, not medical, not a placeholder) and checks whether the true
   patient names are among them.
"""
from __future__ import annotations

import re
from collections import Counter

from ..lexicon import load_lexicon
from ..rules import LABS


def _sex(text: str) -> str:
    m = len(re.findall(r"\b(?:he|his|him|male|man|gentleman|mr)\b", text, re.I))
    f = len(re.findall(r"\b(?:she|her|hers|female|woman|lady|mrs|ms)\b", text, re.I))
    return "M" if m > 1.5 * f else "F" if f > 1.5 * m else "U"


def background_knowledge(anonymizer, record: dict, detection) -> dict:
    """What a well-informed adversary knows about the person behind `record` (from the original)."""
    texts, spans, _, _ = detection
    ages = [s.meta["age"] for s in sorted(spans, key=lambda s: (s.field != "note", s.start)) if s.label == "AGE"]
    occ = [s for s in spans if s.label == "OCCUPATION"]
    locs = sorted({texts[s.field][s.start:s.end].lower() for s in spans if s.label == "LOCATION"})
    return {"age": ages[0] if ages else None, "sex": _sex(record["note"] + " " + record["dialogue"][:3000]),
            "occupation": texts[occ[0].field][occ[0].start:occ[0].end].lower() if occ else None,
            "occupation_category": occ[0].meta.get("category") if occ else None, "locations": locs}


def linkage_attack(anonymizer, records: list[dict], outputs: list[dict], detections: list) -> dict:
    """Compare re-identification risk of the original vs the anonymised release."""
    know = [background_knowledge(anonymizer, r, d) for r, d in zip(records, detections)]
    # released (anonymised) view, parsed from the published text only
    pub = []
    for o in outputs:
        text = o["note"] + " " + o["dialogue"]
        bands = re.findall(r"\[(\d+)-(\d+)\]|\[(\d+)\+\]|\[(<18|18-64|65\+|<1)\]", text)
        band = None
        if bands:
            b = bands[0]
            band = (int(b[0]), int(b[1])) if b[0] else (int(b[2]), 200) if b[2] else \
                {"<18": (0, 17), "18-64": (18, 64), "65+": (65, 200), "<1": (0, 0)}[b[3]]
        occ = re.findall(r"\[OCCUPATION: ([a-z ]+)\]", text)
        pub.append({"band": band, "sex": _sex(o["note"] + " " + o["dialogue"][:3000]), "occ": occ[0] if occ else None,
                    "occ_hidden": "[OCCUPATION" in text, "text": text.lower()})

    def candidates_original(t):
        out = []
        for j, k in enumerate(know):
            if t["age"] is not None and k["age"] != t["age"]:
                continue
            if t["sex"] != "U" and k["sex"] != t["sex"]:
                continue
            if t["occupation"] and k["occupation"] != t["occupation"]:
                continue
            if t["locations"] and not set(t["locations"]) & set(k["locations"]):
                continue
            out.append(j)
        return out

    def candidates_anonymised(t):
        out = []
        for j, p in enumerate(pub):
            if t["age"] is not None and p["band"] and not (p["band"][0] <= t["age"] <= p["band"][1]):
                continue
            if t["sex"] != "U" and p["sex"] != t["sex"]:
                continue
            if t["occupation_category"] and p["occ"] and p["occ"] != t["occupation_category"]:
                continue
            if t["occupation_category"] and not p["occ_hidden"]:
                continue
            out.append(j)
        # a location that survived anonymisation is strong evidence: restrict to records showing it
        leaked = [j for j in out if any(loc in pub[j]["text"] for loc in t["locations"])]
        return leaked or out

    return {name: score_candidates([fn(t) for t in know])
            for name, fn in (("original_release", candidates_original), ("anonymised_release", candidates_anonymised))}


def score_candidates(cand_sets: list[list[int]]) -> dict:
    """Candidate sets indexed by target position -> logically separated outcomes."""
    n = len(cand_sets)
    correct_unique = wrong_unique = no_cand = target_in = 0
    probs, sizes = [], []
    for i, cands in enumerate(cand_sets):
        sizes.append(len(cands))
        if not cands:
            no_cand += 1
            probs.append(0.0)
            continue
        if i in cands:
            target_in += 1
            probs.append(1 / len(cands))
            correct_unique += len(cands) == 1
        else:
            probs.append(0.0)
            wrong_unique += len(cands) == 1
    r = lambda x: round(x / max(1, n), 4)  # noqa: E731
    return {"targets": n, "expected_reidentification": round(sum(probs) / max(1, n), 4),
            "correct_unique_rate": r(correct_unique), "wrong_unique_rate": r(wrong_unique),
            "target_in_candidates_rate": r(target_in), "no_candidate_rate": r(no_cand),
            "mean_candidate_set_size": round(sum(sizes) / max(1, n), 2)}


_LAB_RX = {lab: re.compile(rf"\b(?:{pat})\b[^.?!\n\[]{{0,40}}?(?<![\d.])(\d+(?:\.\d+)?)", re.I) for lab, (pat, _, _) in LABS.items()}


def _lab_values(text: str) -> set:
    out = set()
    for lab, rx in _LAB_RX.items():
        for m in rx.finditer(text):
            out.add((lab, float(m.group(1))))
    return out


def _matches(target: set, released: set, tol: float) -> bool:
    for lab, v in target:
        eps = tol * abs(v) + 1e-9
        if not any(l2 == lab and abs(v2 - v) <= eps for l2, v2 in released):
            return False
    return True


def lab_linkage_attack(records: list[dict], outputs: list[dict], tolerances=(0.0, 0.03, 0.05)) -> dict:
    """Adversary knows each target's exact (lab, value) pairs (e.g. a leaked lab database) and links
    them to released encounters - by exact equality and, more realistically, within a relative
    tolerance that covers a small perturbation. Only targets with >= 1 lab value are counted."""
    targets = [_lab_values(r["note"]) for r in records]
    released_orig = [_lab_values(r["note"] + " " + r["dialogue"]) for r in records]
    released_anon = [_lab_values(o["note"] + " " + o["dialogue"]) for o in outputs]
    idx = [i for i, t in enumerate(targets) if t]
    res = {"targets_with_labs": len(idx)}
    for name, rel in (("original_release", released_orig), ("anonymised_release", released_anon)):
        for tol in tolerances:
            sets = [[j for j in range(len(rel)) if _matches(targets[i], rel[j], tol)] for i in idx]
            # score_candidates expects target i at position i: remap
            remapped = [[idx.index(j) if j in idx else -1 - j for j in s] for s in sets]
            res[f"{name}@tol{tol:.0%}"] = score_candidates(remapped)
    return res


def qi_uniqueness(records: list[dict], outputs: list[dict]) -> dict:
    """Share of encounters that are *unique* in the release on combinations of quasi-identifiers an
    adversary can read off the published text (age band, sex, occupation category) and the ICD-10
    code set, which the task requires to stay unchanged. Original vs anonymised release."""
    def published(o: dict, original: bool) -> dict:
        text = o["note"] + " " + o["dialogue"]
        if original:
            ages = re.findall(r"\b(\d{1,3})[- ]year[- ]old", text)
            age = ages[0] if ages else None
        else:
            bands = re.findall(r"\[(\d+-\d+|\d+\+|<1|<18|18-64|65\+|AGE)\]", text)
            age = bands[0] if bands else None
        occ = re.findall(r"\[OCCUPATION(?:: ([a-z ]+))?\]", text) if not original else []
        return {"age": age, "sex": _sex(o["note"] + " " + o["dialogue"][:3000]),
                "occ": tuple(sorted(set(occ))), "icd": tuple(sorted(c["code"] for c in o.get("icd10_codes", [])))}

    res = {}
    for name, rel, orig in (("original_release", records, True), ("anonymised_release", outputs, False)):
        views = [published(o, orig) for o in rel]
        for combo in (("age", "sex"), ("age", "sex", "occ"), ("icd",), ("age", "sex", "icd")):
            keys = [tuple(v[c] for c in combo) for v in views]
            counts = Counter(keys)
            res[f"{name}:{'+'.join(combo)}"] = round(sum(counts[k] == 1 for k in keys) / max(1, len(keys)), 4)
    return res


def perturbation_inversion_attack(records: list[dict], public_seed: int = 13, max_rel: float = 0.03) -> dict:
    """Can an attacker who knows the published configuration recompute the perturbation and recover
    the exact lab value? Scheme 'public' keys the PRNG with (config seed, encounter id) - both public;
    scheme 'secret' uses an unknown per-run key (current default)."""
    import secrets as _secrets

    from ..rules import RuleDetector
    from ..lexicon import load_lexicon
    from ..transform import perturb_lab

    rules = RuleDetector(load_lexicon())
    n = 0
    recovered = {"public": 0, "secret": 0}
    for r in records:
        eid = str(r.get("encounter_id", ""))
        for s in rules.detect(r["note"], "note"):
            if s.label != "LAB_VALUE":
                continue
            lab, v, dec = s.meta["lab"], s.meta["value"], s.meta["decimals"]
            for scheme in ("public", "secret"):
                key = str(public_seed) if scheme == "public" else _secrets.token_hex(16)
                released = perturb_lab(lab, v, dec, f"{key}|{eid}|{lab}|{v}", max_rel)
                if released is None:
                    continue
                if scheme == "public":
                    n += 1
                # attacker: every original v0 within the tolerance window whose public-key perturbation
                # reproduces the released value
                step = 10 ** -dec
                lo, hi = released * (1 - max_rel) - step, released * (1 + max_rel) + step
                cands = [round(lo + k * step, dec) for k in range(int((hi - lo) / step) + 2)]
                hits = [v0 for v0 in cands if v0 > 0 and perturb_lab(lab, v0, dec, f"{public_seed}|{eid}|{lab}|{v0}", max_rel) == released]
                recovered[scheme] += hits == [v]
    return {"perturbed_values": n,
            "exact_recovery_public_seed": round(recovered["public"] / max(1, n), 4),
            "exact_recovery_secret_key": round(recovered["secret"] / max(1, n), 4)}


def residual_name_attack(records: list[dict], outputs: list[dict]) -> dict:
    """Harvest name-like tokens from the release; success if a true seed name is among them."""
    lex = load_lexicon()
    success = total = 0
    for r, o in zip(records, outputs):
        seeds = {w.lower() for w in re.findall(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-z'\-]+)", r["note"])}
        for m in re.finditer(r"(?:^|\n)((?:[A-Z][a-z'\-]+\s){0,2}[A-Z][a-z'\-]+) is an? \d+-year-old", r["note"]):
            seeds |= {w.lower() for w in m.group(1).split() if w.lower() not in {"the", "patient", "this"}}
        if not seeds:
            continue
        total += 1
        text = o["note"] + "\n" + o["dialogue"]
        harvested = set()
        for m in re.finditer(r"[A-Za-z][a-z'\-]+", text):
            w = m.group(0).lower().removesuffix("'s")
            nxt = re.findall(r"[a-z]+", text[m.end():m.end() + 20].lower())[:2]
            if w in lex.eponyms and any(n in lex.eponyms[w] or n in ("disease", "syndrome", "s") for n in nxt):
                continue  # Parkinson's disease etc. is medical content, not a name candidate
            if (m.group(0)[0].isupper() or lex.name_ratio(w) >= 0.5) and lex.medical_strength(w) == 0 \
                    and lex.zipf(w) < 4.5:
                harvested.add(w)
        success += bool(seeds & harvested)
    return {"records_with_seed_names": total, "name_recovered_rate": round(success / max(1, total), 4)}
