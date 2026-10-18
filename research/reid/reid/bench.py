"""Orchestrator: attacker runs (separate processes) -> evaluation -> JSON results.

    python -m reid.bench attack   --runs D:/.../reid_runs --worlds dev1 dev2 --policies original default age_bands ...
    python -m reid.bench controls --runs ... --world dev1 --policy default --indep-register test1
    python -m reid.bench evaluate --runs ... --worlds dev1 dev2 --out results/dev.json [--pooled]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import clinical, evaluate as ev

HERE = Path(__file__).resolve().parents[1]
THREATS = ("A", "B", "C")


def run_attack(release: Path, register: Path, out: Path, extra=()):
    if (out / "scores.npz").exists():
        return
    cmd = [sys.executable, "-m", "reid.attack.run", "--release", str(release), "--register", str(register), "--out", str(out), *extra]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8")
    print(out, r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else r.stderr[-800:], flush=True)


def cmd_attack(a):
    runs = Path(a.runs)
    for w in a.worlds:
        for pol in a.policies:
            rel = runs / w / "anonymized" / pol / "encounters.jsonl"
            if not rel.exists():
                print("missing", rel)
                continue
            for t in a.threats:
                run_attack(rel, runs / w / "attacker" / f"aux_{t}.parquet", runs / w / "attack" / pol / t)


def cmd_controls(a):
    """Negative/robustness controls on the attacker's knowledge (evaluator prepares the corrupted registers)."""
    runs = Path(a.runs)
    w = runs / a.world
    rel = w / "anonymized" / a.policy / "encounters.jsonl"
    ctl = w / "attacker_controls"
    ctl.mkdir(exist_ok=True)
    rng = np.random.default_rng(99)
    for t in a.threats:
        reg = pd.read_parquet(w / "attacker" / f"aux_{t}.parquet")
        sh = reg.copy()  # every attribute column permuted independently: knowledge unrelated to the person
        for c in sh.columns:
            if c != "key":
                sh[c] = sh[c].sample(frac=1, random_state=int(rng.integers(1 << 30))).to_numpy()
        sh.to_parquet(ctl / f"aux_{t}_shuffled.parquet", index=False)
        inc = reg.copy()  # incomplete knowledge: each known non-name value dropped with probability 0.5
        for c in inc.columns:
            if c not in ("key", "first", "last"):
                mask = rng.random(len(inc)) < 0.5
                inc[c] = inc[c].astype(object).where(~mask, None)
        inc.to_parquet(ctl / f"aux_{t}_incomplete.parquet", index=False)
        run_attack(rel, ctl / f"aux_{t}_shuffled.parquet", w / "attack_controls" / "shuffled_register" / a.policy / t)
        run_attack(rel, ctl / f"aux_{t}_incomplete.parquet", w / "attack_controls" / "incomplete_register" / a.policy / t)
        if a.indep_register:  # register of an independent population: every target is absent
            run_attack(rel, runs / a.indep_register / "attacker" / f"aux_{t}.parquet",
                       w / "attack_controls" / f"indep_{a.indep_register}" / a.policy / t)
        run_attack(rel, w / "attacker" / f"aux_{t}.parquet", w / "attack_controls" / "demographics_only" / a.policy / t,
                   extra=("--attrs", "sex,by"))


def _eval_one(world: Path, scores: Path, t: str, truth, persons, world_types=("closed", "open"), pools=ev.POOLS,
              shuffle=False):
    out = {}
    for wt in world_types:
        for P in pools:
            df = ev.outcomes(world, scores, t, P, wt, truth, persons, shuffle_truth=shuffle)
            out[f"{wt}/{P}"] = {m: ev.metrics(df, m, P, wt, (world.name, t)) for m in ("fs", "det")}
            out[f"{wt}/{P}"]["_frame"] = df
    return out


PAIRS = (("original", "default"), ("default", "age_bands"), ("age_bands", "age_bands_strict_dates"),
         ("default", "age_bands_strict_dates"))


def paired(frames: dict) -> dict:
    """Paired policy differences on identical queries and pools: success(a) - success(b), patient-cluster bootstrap."""
    out = {}
    for pa, pb in PAIRS:
        for (pol, t, k), fl in frames.items():
            if pol != pa or (pb, t, k) not in frames or k not in ("closed/2000", "open/2000", "closed/100"):
                continue
            A_ = pd.concat(fl, ignore_index=True).set_index(["world", "rid"])
            B_ = pd.concat(frames[(pb, t, k)], ignore_index=True).set_index(["world", "rid"]).reindex(A_.index)
            res = {}
            for m in ("fs", "det"):
                c = f"{m}_correct"
                if c not in A_ or c not in B_:
                    continue
                d = pd.DataFrame({"pid": A_["pid"].to_numpy(), "a": A_[c].fillna(False).astype(float).to_numpy(),
                                  "b": B_[c].fillna(False).astype(float).to_numpy(), "n": A_["present"].astype(float).to_numpy()})
                g = d.groupby("pid")[["a", "b", "n"]].sum()
                diff = (g["a"] - g["b"]).to_numpy()
                n = g["n"].to_numpy()
                rng = np.random.default_rng(11)
                idx = rng.integers(0, len(g), size=(ev.B, len(g)))
                bs = diff[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
                res[m] = {"success_a": round(g["a"].sum() / n.sum(), 4), "success_b": round(g["b"].sum() / n.sum(), 4),
                          "diff": round(diff.sum() / n.sum(), 4), "ci": [round(float(np.percentile(bs, 2.5)), 4),
                                                                          round(float(np.percentile(bs, 97.5)), 4)],
                          "discordant_a_only": int(((d["a"] > 0) & (d["b"] == 0)).sum()),
                          "discordant_b_only": int(((d["b"] > 0) & (d["a"] == 0)).sum())}
            out[f"{pa} vs {pb} | {t} | {k}"] = res
    return out


def cmd_evaluate(a):
    runs = Path(a.runs)
    res = {"tau": ev.TAU, "pools": ev.POOLS, "bootstrap": ev.B, "worlds": {}}
    frames = {}
    t0 = time.time()
    for wn in a.worlds:
        w = runs / wn
        truth, persons, man = ev.load_world(w)
        R = res["worlds"][wn] = {"manifest": {k: man[k] for k in ("seed", "family", "n_encounters", "n_patients",
                                                                  "population_with_doppelgangers", "n_hard_targets")}}
        for pol_dir in sorted((w / "attack").glob("*")):
            pol = pol_dir.name
            P_ = R.setdefault(pol, {})
            out_file = w / "anonymized" / pol / "encounters.jsonl"
            P_["clinical"] = clinical.check(w, out_file, persons)
            for t in THREATS:
                sc = pol_dir / t / "scores.npz"
                if not sc.exists():
                    continue
                info = json.loads((pol_dir / t / "attack.json").read_text())
                e = _eval_one(w, sc, t, truth, persons)
                for k, v in e.items():
                    frames.setdefault((pol, t, k), []).append(v.pop("_frame").assign(world=wn))
                P_[t] = {"metrics": e, "inference": ev.inference(w, sc, t, truth),
                         "attack": {k: info.get(k) for k in ("attrs", "feature_coverage", "total_s")} |
                                   {"splink": {k: info.get("splink", {}).get(k) for k in ("sessions", "neutralised_levels", "train_s", "predict_s")},
                                    "gamma_agreement_min": min(info.get("gamma_agreement_numpy_vs_splink", {"-": 1}).values())}}
                if pol == a.shuffle_policy:  # negative control: shuffled record->patient mapping
                    P_[t]["shuffled_truth"] = {k: {m: v[m] for m in ("fs", "det")} for k, v in
                                               _eval_one(w, sc, t, truth, persons, ("closed",), (100, 2000), shuffle=True).items()}
        ctl_root = w / "attack_controls"
        for ctl in sorted(ctl_root.glob("*/*/*")) if ctl_root.exists() else []:
            name, pol, t = ctl.parts[-3:]
            if not (ctl / "scores.npz").exists():
                continue
            if name.startswith("indep_"):  # independent population: all targets absent, keys unknown to truth
                z = np.load(ctl / "scores.npz", allow_pickle=True)
                L = z["log2bf"]
                g = np.random.default_rng(5)
                fm = {}
                for P in (100, 2000):
                    ids = 0
                    for q in range(L.shape[0]):
                        cols = g.choice(L.shape[1], P, replace=False)
                        s = L[q, cols].astype(float)
                        m = s.max()
                        post = 1.0 / (np.exp2(s - m).sum() + P * np.exp2(-m))
                        ids += post >= ev.TAU
                    fm[f"open/{P}"] = round(ids / L.shape[0], 4)
                R.setdefault("controls", {}).setdefault(name, {}).setdefault(pol, {})[t] = {"fs_false_match_rate": fm}
            else:
                e = _eval_one(w, ctl / "scores.npz", t, truth, persons, ("closed", "open"), (100, 2000))
                R.setdefault("controls", {}).setdefault(name, {}).setdefault(pol, {})[t] = {
                    k: {m: v[m] for m in ("fs", "det")} for k, v in e.items()}
    if a.pooled and len(a.worlds) > 1:  # pooled across worlds; bootstrap clusters = patients (shared across worlds)
        res["pooled"] = {}
        for (pol, t, k), fl in frames.items():
            wt, P = k.split("/")
            df = pd.concat(fl, ignore_index=True)
            res["pooled"].setdefault(pol, {}).setdefault(t, {})[k] = {m: ev.metrics(df, m, int(P), wt, ("pooled", t)) for m in ("fs", "det")}
    res["paired"] = paired(frames)
    res["eval_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print("wrote", a.out, res["eval_s"], "s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("attack")
    p.add_argument("--runs", required=True)
    p.add_argument("--worlds", nargs="+", required=True)
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--threats", nargs="+", default=list(THREATS))
    p = sub.add_parser("controls")
    p.add_argument("--runs", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--threats", nargs="+", default=list(THREATS))
    p.add_argument("--indep-register")
    p = sub.add_parser("evaluate")
    p.add_argument("--runs", required=True)
    p.add_argument("--worlds", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pooled", action="store_true")
    p.add_argument("--shuffle-policy", default="default")
    a = ap.parse_args(argv)
    {"attack": cmd_attack, "controls": cmd_controls, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
