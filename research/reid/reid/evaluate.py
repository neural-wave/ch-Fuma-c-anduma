"""Evaluator (sees truth). Turns attacker pairwise scores into pool-level linkage outcomes and metrics.

Pools: for every target patient and pool size P, decoys are drawn from the population register (seeded by world,
P and patient, identical for every policy and threat model); the patient's doppelgangers (hard negatives) are always
in the pool. Closed world: the patient is in the pool. Open world: the patient is present with probability 0.5
(fixed per patient), otherwise the pool holds P decoys.
Attackers:
  fs     Fellegi-Sunter (Splink m/u): pool posterior BF_j / ((1-pi)/pi * P + sum_k BF_k); identify the top candidate if
         posterior >= tau (tau fixed a priori, not tuned on truth); pi = 1 closed, 0.5 open (attacker's assumption).
  det    deterministic: candidates compatible on every published attribute; identify only a unique compatible
         candidate with >= 2 agreeing attributes; rank by (compatible, n_agree).
  random analytic: top-1 = 1/P, top-5 = 5/P, MRR = H_P / P.
Ties in ranks are broken uniformly at random (seeded). Uncertainty: percentile bootstrap over patients (B=1000).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOLS = (100, 500, 2000, 10000)
TAU = 0.9
B = 1000


def _rng(*parts) -> np.random.Generator:
    import hashlib
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16))


def load_world(world: Path):
    truth = [json.loads(line) for line in open(world / "private" / "encounters_truth.jsonl", encoding="utf-8")]
    with gzip.open(world / "private" / "persons.csv.gz", "rt", encoding="utf-8") as f:
        persons = pd.read_csv(f, low_memory=False)
    man = json.loads((world / "private" / "manifest.json").read_text())
    return truth, persons, man


def pool_posterior(log2bf: np.ndarray, P: int, pi: float) -> np.ndarray:
    """Posterior that candidate j is the record's person, given presence prior pi and a uniform prior over the pool:
    BF_j / ((1 - pi) / pi * P + sum_k BF_k)  (computed in log space)."""
    s = np.asarray(log2bf, dtype=np.float64)
    m = s.max()
    w = np.exp2(s - m)
    return w / (w.sum() + ((1 - pi) / pi) * P * np.exp2(-m))


def build_pools(world: Path, truth, persons, P: int, shuffle_truth: bool = False):
    """{pid: (decoy_pids, present_open)} identical across policies/threats for a given world and P."""
    seed = json.loads((world / "private" / "manifest.json").read_text())["seed"]
    pids_all = persons["pid"].to_numpy()
    dop = persons.dropna(subset=["doppelganger_of"]).groupby("doppelganger_of")["pid"].apply(list).to_dict()
    pools = {}
    for t in sorted({x["pid"] for x in truth}):
        g = _rng(seed, P, t)
        hard = dop.get(t, [])
        excl = set(hard) | {t}
        need = P - len(hard)  # open-absent pools hold P decoys; closed pools replace one decoy by the target
        idx = g.choice(len(pids_all), size=min(need + len(excl) + 5, len(pids_all)), replace=False)
        dec = [p for p in pids_all[idx] if p not in excl][:need]
        present = bool(_rng(seed, "presence", t).random() < 0.5)
        pools[t] = (hard + dec, present)
    return pools


def outcomes(world: Path, scores: Path, threat: str, P: int, world_type: str, truth, persons, shuffle_truth=False):
    z = np.load(scores, allow_pickle=True)
    rid_idx = {r: i for i, r in enumerate(z["rids"])}
    keys = pd.read_csv(world / "private" / f"keys_{threat}.csv")
    col_of = dict(zip(keys["pid"], pd.Series(range(len(z["keys"])), index=z["keys"]).reindex(keys["key"]).to_numpy()))
    L, C, N = z["log2bf"], z["compat"], z["n_agree"]
    pools = build_pools(world, truth, persons, P)
    tpids = sorted({x["pid"] for x in truth})
    relabel = dict(zip(tpids, _rng("shuffle", P).permutation(tpids))) if shuffle_truth else {p: p for p in tpids}
    rows = []
    for x in truth:
        q = rid_idx[x["encounter_id"]]
        t = relabel[x["pid"]]
        dec, present = pools[t]
        present = True if world_type == "closed" else present
        cand = ([t] + dec[:P - 1]) if present else dec[:P]
        cols = np.array([col_of[p] for p in cand])
        g = _rng("tie", world.name, threat, P, world_type, x["encounter_id"])
        jit = g.random(len(cols)) * 1e-6
        s = L[q, cols].astype(np.float64)
        row = {"rid": x["encounter_id"], "pid": x["pid"], "present": present, "hard": bool(len(dec) and str(dec[0]).startswith("dup"))}
        # ---- Fellegi-Sunter
        if not np.isnan(s).all():
            order = np.argsort(-(s + jit))
            post = pool_posterior(s, P, 1.0 if world_type == "closed" else 0.5)[order[0]]
            row.update(fs_rank=int(np.where(order == 0)[0][0]) + 1 if present else None,
                       fs_id=bool(post >= TAU), fs_correct=bool(post >= TAU and present and order[0] == 0),
                       fs_post=float(post))
        # ---- deterministic
        comp, nag = C[q, cols], N[q, cols].astype(np.float64)
        key = comp * 100 + nag + jit
        order = np.argsort(-key)
        uniq = comp.sum() == 1
        best = order[0]
        det_id = bool(uniq and comp[best] and nag[best] >= 2)
        row.update(det_rank=int(np.where(order == 0)[0][0]) + 1 if present else None, det_id=det_id,
                   det_correct=bool(det_id and present and best == 0), det_ncompat=int(comp.sum()),
                   det_true_compat=bool(comp[0]) if present else None)
        rows.append(row)
    return pd.DataFrame(rows)


def _ci(per_pat_num: np.ndarray, per_pat_den: np.ndarray, g: np.random.Generator):
    n = len(per_pat_num)
    if n == 0 or per_pat_den.sum() == 0:
        return None, None, None
    est = per_pat_num.sum() / per_pat_den.sum()
    idx = g.integers(0, n, size=(B, n))
    num, den = per_pat_num[idx].sum(1), per_pat_den[idx].sum(1)
    bs = num[den > 0] / den[den > 0]
    return round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)


def metrics(df: pd.DataFrame, method: str, P: int, world_type: str, seed_parts=()) -> dict:
    g = _rng("boot", method, P, world_type, *seed_parts)
    df = df.copy()
    pat = df.groupby("pid")
    out = {"n_queries": len(df), "n_patients": df["pid"].nunique()}

    def ratio(name, num_col, den_mask_col=None):
        d = df.assign(_num=df[num_col].astype(float), _den=1.0 if den_mask_col is None else df[den_mask_col].astype(float))
        s = d.groupby("pid")[["_num", "_den"]].sum()
        out[name] = _ci(s["_num"].to_numpy(), s["_den"].to_numpy(), g)

    idc, cor = f"{method}_id", f"{method}_correct"
    if idc not in df:
        return out
    df["wrong"] = df[idc] & ~df[cor]
    df["absent"] = ~df["present"]
    pres = df[df["present"]]
    if len(pres) and pres[f"{method}_rank"].notna().any():
        rk = pres[f"{method}_rank"].astype(float)
        df.loc[pres.index, "top1"] = (rk == 1).astype(float)
        df.loc[pres.index, "top5"] = (rk <= 5).astype(float)
        df.loc[pres.index, "rr"] = 1.0 / rk
        for c in ("top1", "top5", "rr"):
            df[c] = df[c].fillna(0.0)
        ratio("top1", "top1", "present")
        ratio("top5", "top5", "present")
        ratio("mrr", "rr", "present")
    ratio("success_over_eligible", cor, "present")
    ratio("coverage", idc)
    out["precision"] = _ci(*(lambda s: (s[cor].to_numpy(float), s[idc].to_numpy(float)))(df.groupby("pid")[[cor, idc]].sum()), g)
    ratio("false_identification_rate", "wrong")
    if world_type == "open":
        df["id_absent"] = df[idc] & df["absent"]
        ratio("false_match_rate_absent", "id_absent", "absent")
        d = df[df["absent"]]
        out["false_match_rate_absent_hard"] = round(float(d[d["hard"]][idc].mean()), 4) if d["hard"].any() else None
    # patient level: a present patient counts as re-identified if any of their encounters is correctly identified
    pp = pres.groupby("pid")[cor].any().to_numpy(float) if len(pres) else np.array([])
    out["patient_reidentified"] = _ci(pp, np.ones_like(pp), g) if len(pp) else None
    out["patient_any_false_id"] = round(float(df.groupby("pid")["wrong"].any().mean()), 4)
    out["random_top1"], out["random_top5"] = round(1 / P, 5), round(min(5, P) / P, 5)
    out["random_mrr"] = round(float(np.sum(1 / np.arange(1, P + 1)) / P), 5)
    if "hard" in df and df["hard"].any() and len(pres):
        h = pres[pres["hard"]]
        out["hard_success"] = round(float(h[cor].mean()), 4) if len(h) else None
        out["hard_n"] = int(len(h))
    return out


def inference(world: Path, scores: Path, threat: str, truth) -> dict:
    """Identity -> record attribute inference: for each target, the posterior over released records (target known to
    have >= 1 record) votes for the primary-diagnosis ICD-10 chapter. Baselines: prior-only mode, random record."""
    z = np.load(scores, allow_pickle=True)
    L = z["log2bf"]
    if np.isnan(L).all():
        return {}
    rid_idx = {r: i for i, r in enumerate(z["rids"])}
    keys = pd.read_csv(world / "private" / f"keys_{threat}.csv")
    kcol = pd.Series(range(len(z["keys"])), index=z["keys"])
    col_of = dict(zip(keys["pid"], kcol.reindex(keys["key"]).to_numpy()))
    chap = np.empty(len(rid_idx), dtype=object)
    for x in truth:
        chap[rid_idx[x["encounter_id"]]] = x["primary_icd"][0]
    by_pid = {}
    for x in truth:
        by_pid.setdefault(x["pid"], set()).add(x["primary_icd"][0])
    mode = pd.Series(chap).value_counts().index[0]
    hit = prior = rnd = 0.0
    for pid, chs in by_pid.items():
        s = L[:, col_of[pid]].astype(np.float64)
        w = np.exp2(s - s.max())
        votes = pd.Series(w).groupby(chap).sum()
        hit += votes.idxmax() in chs
        prior += mode in chs
        rnd += np.mean([c in chs for c in chap])
    n = len(by_pid)
    return {"targets": n, "attack_acc": round(hit / n, 4), "prior_only_acc": round(prior / n, 4),
            "random_record_acc": round(rnd / n, 4)}
