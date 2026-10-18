"""Attacker process. Sees ONLY an anonymised release and one attacker register; writes pairwise scores.

    python -m reid.attack.run --release <anonymized.jsonl> --register <aux_T.parquet> --out <dir> [--seed 7] [--params P]

Outputs <out>/scores.npz (rids, keys, log2bf [Fellegi-Sunter log2 Bayes factor from Splink], compat and n_agree
[deterministic baseline]) and <out>/attack.json (active comparisons, trained m/u, Splink-vs-numpy gamma agreement,
timings). Refuses inputs under private/ or release_input/ (the evaluator's directories).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import active, deterministic, levels_numpy, splink_comparisons
from .extract import Vocab, extract, load_common_words, load_occupation_categories, register_features

LIST_COLS = ("nm_found", "nm", "hb", "ev", "icd", "vis_y", "vis_ym")
COLS = ("nm_found", "nm", "sex", "by_lo", "by_hi", "dob_md", "city", "occ_title", "occ_cat", "marital", "kids_lo", "kids_hi",
        "work", "hb", "race", "ht_lo", "ht_hi", "wt_lo", "wt_hi", "bmi_lo", "bmi_hi", "ev", "icd", "vis_y", "vis_ym")


def _guard(p: str) -> Path:
    parts = {x.lower() for x in Path(p).resolve().parts}
    if parts & {"private", "release_input"}:
        raise SystemExit(f"attacker refuses evaluator-side input: {p}")
    return Path(p)


def features(release_path: Path, register_path: Path):
    recs = [json.loads(line) for line in open(release_path, encoding="utf-8")]
    reg = pd.read_parquet(register_path)
    cats = load_occupation_categories()
    voc = Vocab(reg, cats, load_common_words())
    R = pd.DataFrame([extract(r, voc) for r in recs])
    A = register_features(reg, cats)
    for df in (R, A):
        for c in COLS:
            if c not in df:
                df[c] = None
    return R, A


def _typed(df: pd.DataFrame, uid: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"uid": uid.astype(str)})
    for c in COLS:
        s = df[c]
        if c in LIST_COLS:
            out[c] = [None if v is None or (isinstance(v, float)) else [str(x) for x in v] for v in s]
        elif c.endswith(("_lo", "_hi")):
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[c] = s.astype(object).where(s.notna(), None).astype("string")
    return out


def _neutralise_unobserved(linker) -> list[str]:
    """A level never seen in any EM session gets Splink's 'not observed' m (1e-6), i.e. a near-veto. Treat it as
    no evidence instead (m = u): conservative for the attacker, and logged in attack.json."""
    from splink.internals.constants import LEVEL_NOT_OBSERVED_TEXT as NO

    out = []
    for comp in linker._settings_obj.comparisons:
        for lv in comp.comparison_levels:
            if lv.is_null_level:
                continue
            m, u = lv._m_probability, lv._u_probability
            if m in (None, NO) or u in (None, NO):
                base = u if u not in (None, NO) else (m if m not in (None, NO) else 0.5)
                lv.m_probability = lv.u_probability = base
                out.append(f"{comp.output_column_name}:{lv.label_for_charts}")
    return out


def splink_scores(R: pd.DataFrame, A: pd.DataFrame, attrs: list[str], seed: int, params: dict | None = None):
    from splink import Linker, SettingsCreator
    from splink.backends.duckdb import DuckDBAPI

    logging.getLogger("splink").setLevel(logging.WARNING)
    db = DuckDBAPI()
    left = db.register(_typed(R, "r" + R.index.astype(str)), dataset_display_name="release")
    right = db.register(_typed(A, "a" + A.index.astype(str)), dataset_display_name="register")
    lam = 1.0 / len(A)
    settings = SettingsCreator(link_type="link_only", comparisons=splink_comparisons(attrs),
                               blocking_rules_to_generate_predictions=["1=1"], probability_two_random_records_match=lam,
                               unique_id_column_name="uid", retain_intermediate_calculation_columns=True,
                               max_iterations=50)
    linker = Linker([left, right], settings, log_level=logging.WARNING)
    t0 = time.time()
    linker.training.estimate_u_using_random_sampling(max_pairs=3e6, seed=seed)
    sessions = []
    if params is None:
        rules = []
        if "sex" in attrs and "by" in attrs:
            rules.append("l.sex = r.sex AND l.by_lo <= r.by_hi AND r.by_lo <= l.by_hi")
        second = [("icd", "len(list_intersect(l.icd, r.icd)) >= 1"), ("ev", "len(list_intersect(l.ev, r.ev)) >= 1"),
                  ("occ", "l.occ_cat = r.occ_cat"), ("kids", "l.kids_lo = r.kids_lo AND l.kids_hi = r.kids_hi"),
                  ("race", "l.race = r.race AND l.sex = r.sex")]
        extra = [sql for a, sql in second if a in attrs][:1]
        if not extra and "sex" in attrs and "by" in attrs:  # demographics only: estimate each from the other
            rules = ["l.sex = r.sex", "l.by_lo <= r.by_hi AND r.by_lo <= l.by_hi AND greatest(l.by_hi - l.by_lo, r.by_hi - r.by_lo) <= 3"]
        rules += extra
        snaps = [linker.misc.save_model_to_json()]
        for br in rules:
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(
                    br, fix_probability_two_random_records_match=True)
                sessions.append({"blocking": br, "ok": True})
                snaps.append(linker.misc.save_model_to_json())
            except Exception as e:  # noqa: BLE001 - record and continue with remaining sessions
                sessions.append({"blocking": br, "ok": False, "error": str(e)[:300]})
        have = {c.output_column_name for c in linker._settings_obj.comparisons}
        if have != set(attrs):  # Splink 5.0.0 can drop a comparison after an EM session: rebuild from merged snapshots
            merged = json.loads(json.dumps(snaps[0]))
            for c in merged["comparisons"]:
                for snap in snaps[1:]:
                    src = next((x for x in snap["comparisons"] if x["output_column_name"] == c["output_column_name"]), None)
                    if src is None:
                        continue
                    for lv, sl in zip(c["comparison_levels"], src["comparison_levels"]):
                        if "m_probability" in sl:
                            lv["m_probability"] = sl["m_probability"]
            sessions.append({"rebuilt_from_snapshots": sorted(set(attrs) - have)})
            linker = Linker([left, right], merged, log_level=logging.WARNING)
    neutral = _neutralise_unobserved(linker)
    t_train = time.time() - t0
    pred = linker.inference.predict().as_pandas_dataframe()
    t_pred = time.time() - t0 - t_train
    l_is_r = pred["uid_l"].str.startswith("r")
    ri = np.where(l_is_r, pred["uid_l"].str[1:], pred["uid_r"].str[1:]).astype(int)
    ai = np.where(l_is_r, pred["uid_r"].str[1:], pred["uid_l"].str[1:]).astype(int)
    prior = math.log2(lam / (1 - lam))
    L = np.full((len(R), len(A)), np.nan, dtype=np.float32)
    L[ri, ai] = (pred["match_weight"].to_numpy() - prior).astype(np.float32)
    gam = {}
    for a in attrs:
        G = np.full((len(R), len(A)), -2, dtype=np.int8)
        G[ri, ai] = pred[f"gamma_{a}"].to_numpy().astype(np.int8)
        gam[a] = G
    model = json.loads(json.dumps(linker.misc.save_model_to_json() if hasattr(linker, "misc") else {}, default=str))
    mu = {}
    for c in model.get("comparisons", []):
        mu[c["output_column_name"]] = [{"label": lv.get("label_for_charts"), "m": lv.get("m_probability"),
                                         "u": lv.get("u_probability")} for lv in c["comparison_levels"]
                                        if not lv.get("is_null_level")]
    return L, gam, {"sessions": sessions, "neutralised_levels": neutral, "m_u": mu, "lambda": lam, "train_s": round(t_train, 1),
                    "predict_s": round(t_pred, 1), "pairs": int(len(pred))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--register", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attrs", help="comma list to restrict comparisons (ablation)")
    ap.add_argument("--no-splink", action="store_true")
    a = ap.parse_args(argv)
    rel, reg, out = _guard(a.release), _guard(a.register), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R, A = features(rel, reg)
    attrs = active(R, A)
    if a.attrs:
        attrs = [x for x in attrs if x in a.attrs.split(",")]
    lv = levels_numpy(R, A, attrs)
    compat, n_agree = deterministic(lv)
    info = {"release": str(rel), "register": str(reg), "attrs": attrs, "n_records": len(R), "n_register": len(A),
            "feature_coverage": {c: round(float(R[c].notna().mean()), 3) for c in COLS if c in R},
            "extract_s": round(time.time() - t0, 1)}
    log2bf = np.full(compat.shape, np.nan, dtype=np.float32)
    if not a.no_splink and attrs:
        log2bf, gam, sp = splink_scores(R, A, attrs, a.seed)
        info["splink"] = sp
        info["gamma_agreement_numpy_vs_splink"] = {k: round(float((gam[k] == lv[k]).mean()), 6) for k in attrs}
    np.savez_compressed(out / "scores.npz", rids=R["rid"].to_numpy(), keys=A["key"].to_numpy(), log2bf=log2bf,
                        compat=compat, n_agree=n_agree)
    R.to_json(out / "release_features.jsonl", orient="records", lines=True)
    info["total_s"] = round(time.time() - t0, 1)
    (out / "attack.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("attrs", "n_records", "n_register", "total_s")}
                     | {"gamma_agreement": info.get("gamma_agreement_numpy_vs_splink")}))


if __name__ == "__main__":
    main()
