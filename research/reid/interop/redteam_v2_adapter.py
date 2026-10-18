"""Shared evaluation interface: run the teammate's Red Team V2 attacker (read-only import, unmodified) on this
benchmark's anonymised releases, registers and pools, and score both attackers with the same evaluator.

    python interop/redteam_v2_adapter.py --rt2 <MedAnon-RedTeam-V2 dir> --runs <reid_runs> --worlds dev1 dev2 \
        --policies default age_bands age_bands_strict_dates --threats B C --out results/interop_dev.json

Contract (both attackers): input = anonymised MedAnon JSONL + attacker register (opaque keys); output = per record a
ranking of register keys with scores and an identify/abstain decision. Truth, pools and metrics stay in the evaluator.
Adapter glue (documented, no change to RT2 code): register rows -> RT2 AuxiliaryProfile features (age interval as of
2026, sex, (sector, title), geography path, children, hobby as context, visit year as administrative date for C,
ICD-10 3-character codes for C); RT2 extracted ICD codes truncated to 3 characters to match the register.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from reid import evaluate as ev  # noqa: E402


def load_rt2(path: Path):
    sys.path.insert(0, str(path / "src"))
    from medanon.evaluation import adaptive_linkage as al  # teammate code, imported read-only
    from medanon.lexicon import load_lexicon
    return al, load_lexicon().occupations


def profiles(reg: pd.DataFrame, threat: str, al, occ_map) -> dict:
    out = {}
    for r in reg.itertuples(index=False):
        f = {}
        g = lambda c: getattr(r, c, None)  # noqa: E731
        if isinstance(g("birth_date"), str):
            y = int(g("birth_date")[:4])
            f["age"] = (2026 - y - 1, 2026 - y)
        elif g("birth_year") == g("birth_year") and g("birth_year") is not None:
            f["age"] = (2026 - int(g("birth_year")) - 1, 2026 - int(g("birth_year")))
        elif g("age_approx") == g("age_approx") and g("age_approx") is not None:
            f["age"] = (int(g("age_approx") - g("age_tol")), int(g("age_approx") + g("age_tol")))
        if isinstance(g("sex"), str):
            f["sex"] = g("sex")
        if isinstance(g("occupation"), str):
            f["occupation"] = (occ_map.get(g("occupation"), "other"), g("occupation"))
        if isinstance(g("city"), str):
            f["geography"] = al.geographic_path(g("city"))
        if g("children") == g("children") and g("children") is not None:
            f["family_information"] = (int(g("children")), int(g("children")))
        if isinstance(g("hobby"), str):
            f["rare_nonclinical_context"] = g("hobby")
        if threat == "C":
            v = g("visit_ym")
            if v is not None and len(v):
                f["administrative_dates"] = int(v[0][:4])
            icd = g("icd3")
            if icd is not None and len(icd):
                f["optional_medical_knowledge"] = tuple(icd)
        out[r.key] = al.AuxiliaryProfile(r.key, f, {})
    return out


def run(a):
    al, occ_map = load_rt2(Path(a.rt2))
    runs = Path(a.runs)
    th = al.LinkageThreshold(a.score, a.margin, 2)
    res = {"threshold": [a.score, a.margin, 2], "error_rate": 0.1, "pools": a.pools}
    for wn in a.worlds:
        w = runs / wn
        truth, persons, _ = ev.load_world(w)
        for t in a.threats:
            reg = pd.read_parquet(w / "attacker" / f"aux_{t}.parquet")
            prof = profiles(reg, t, al, occ_map)
            kmap = pd.read_csv(w / "private" / f"keys_{t}.csv")
            key_of = dict(zip(kmap["pid"], kmap["key"]))
            for pol in a.policies:
                recs = {r["encounter_id"]: r for r in map(json.loads, open(w / "anonymized" / pol / "encounters.jsonl", encoding="utf-8"))}
                fx = al.PopulationFeatureExtractor()
                feats = {}
                for eid, r in recs.items():
                    f = fx.extract_features(r, diagnosis_knowledge=(t == "C"))
                    if "optional_medical_knowledge" in f:
                        f["optional_medical_knowledge"] = tuple(sorted({c[:3] for c in f["optional_medical_knowledge"]}))
                    feats[eid] = f
                cov = pd.Series([g for f in feats.values() for g in f]).value_counts().div(len(feats)).round(3).to_dict()
                for wt in ("closed", "open"):
                    for P in a.pools:
                        pools = ev.build_pools(w, truth, persons, P)
                        rows = []
                        for x in truth:
                            dec, present = pools[x["pid"]]
                            present = True if wt == "closed" else present
                            cand = ([x["pid"]] + dec[:P - 1]) if present else dec[:P]
                            keys = [key_of[p] for p in cand]
                            att = al.PopulationAttacker([prof[k] for k in keys], error_rate=0.1)
                            pred = att.rank(feats[x["encounter_id"]], th)
                            # full ranking for the true key: recompute scores over the pool (RT2 returns top-5 only)
                            contrib, agree = att.evidence(feats[x["encounter_id"]])
                            sc = sum(contrib.values(), np.zeros(len(keys)))
                            if "age" in contrib and "family_information" in contrib:
                                sc -= 0.5 * np.minimum(np.maximum(contrib["age"], 0), np.maximum(contrib["family_information"], 0))
                            order_keys = [r.key for r in att.references]
                            true_score = sc[order_keys.index(keys[0])] if present else None
                            g = ev._rng("tie-rt2", wn, t, P, wt, x["encounter_id"])
                            rank = (1 + int((sc > true_score).sum()) + int(g.integers(0, max(1, int((sc == true_score).sum()))))) if present else None
                            ident = pred.identified
                            rows.append({"rid": x["encounter_id"], "pid": x["pid"], "present": present,
                                         "hard": bool(len(dec) and str(dec[0]).startswith("dup")),
                                         "rt2_rank": rank, "rt2_id": ident is not None,
                                         "rt2_correct": bool(ident is not None and present and ident == keys[0])})
                        df = pd.DataFrame(rows)
                        res.setdefault(wn, {}).setdefault(pol, {}).setdefault(t, {})[f"{wt}/{P}"] = ev.metrics(df, "rt2", P, wt, (wn, t))
                res[wn][pol][t]["feature_coverage"] = cov
                print(wn, pol, t, {k: v["success_over_eligible"] for k, v in res[wn][pol][t].items() if "/" in k}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt2", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--worlds", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--threats", nargs="+", default=["B", "C"])
    ap.add_argument("--pools", nargs="+", type=int, default=[2000])
    ap.add_argument("--score", type=float, default=4.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
