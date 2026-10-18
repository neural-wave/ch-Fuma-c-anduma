"""Pairwise comparison levels between release features (R) and register features (A).

Two independent implementations of the SAME level coding: numpy (deterministic baseline, hand-checked in tests)
and DuckDB SQL for Splink. Coding follows Splink gammas: -1 = no information (null), 0 = disagreement,
higher = stronger agreement. "Value not published" is always null, never disagreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NARROW = {"ht": 3.0, "wt": 5.0, "bmi": 1.0}
SCALAR = ("sex", "dob_md", "city", "marital", "work", "race")
INTERVAL3 = ("ht", "wt", "bmi")
LISTCOUNT = ("ev", "icd")
ALL = ("names", "sex", "by", "dob_md", "city", "occ", "marital", "kids", "hobby", "work", "race", "ht", "wt", "bmi",
       "ev", "icd", "visit")


def _isnull(s: pd.Series) -> np.ndarray:
    return s.isna().to_numpy() if s.dtype != object else np.array([v is None or (isinstance(v, float) and v != v) for v in s])


def _f(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _listset(s: pd.Series) -> list[set]:
    return [set(v) if v is not None and not (isinstance(v, float)) and len(v) else set() for v in s]


def active(R: pd.DataFrame, A: pd.DataFrame) -> list[str]:
    """Attributes with information on both sides (otherwise the comparison is always null)."""
    def has(df, cols):
        return any(c in df and (~pd.Series(_isnull(df[c]))).any() for c in cols)
    cols = {"names": (["nm_found"], ["nm"]), "by": (["by_lo"], ["by_lo"]), "occ": (["occ_title", "occ_cat"], ["occ_title"]),
            "kids": (["kids_lo"], ["kids_lo"]), "hobby": (["hb"], ["hb"]), "visit": (["vis_y"], ["vis_y"])}
    out = []
    for a in ALL:
        rc, ac = cols.get(a, ([f"{a}_lo"], [f"{a}_lo"]) if a in INTERVAL3 else ([a], [a]))
        if has(R, rc) and has(A, ac):
            out.append(a)
    return out


def levels_numpy(R: pd.DataFrame, A: pd.DataFrame, attrs) -> dict[str, np.ndarray]:
    nr, na = len(R), len(A)
    out = {}
    for a in attrs:
        L = np.full((nr, na), -1, dtype=np.int8)
        if a == "names":
            first = np.array([n[0] for n in A["nm"]])
            last = np.array([n[1] for n in A["nm"]])
            for i, S in enumerate(_listset(R["nm_found"])):
                if S:
                    fi, la = np.isin(first, list(S)), np.isin(last, list(S))
                    L[i] = np.where(fi & la, 3, np.where(la, 2, np.where(fi, 1, 0)))
        elif a in SCALAR:
            rv, av = R[a].to_numpy(dtype=object), A[a].to_numpy(dtype=object)
            rn, an = _isnull(R[a]), _isnull(A[a])
            eq = rv[:, None] == av[None, :]
            L = np.where(rn[:, None] | an[None, :], -1, eq.astype(np.int8)).astype(np.int8)
        elif a in ("by", "kids") or a in INTERVAL3:
            p = {"by": "by", "kids": "kids"}.get(a, a)
            rl, rh, al, ah = _f(R[f"{p}_lo"]), _f(R[f"{p}_hi"]), _f(A[f"{p}_lo"]), _f(A[f"{p}_hi"])
            null = np.isnan(rl)[:, None] | np.isnan(al)[None, :]
            ov = (rl[:, None] <= ah[None, :]) & (al[None, :] <= rh[:, None])
            w = np.maximum((rh - rl)[:, None], (ah - al)[None, :])
            if a == "by":
                L = np.select([null, ov & (w <= 1), ov & (w <= 3), ov & (w <= 6), ov], [-1, 4, 3, 2, 1], 0)
            elif a == "kids":
                ex = ((rl == rh)[:, None] & (al == ah)[None, :] & (rl[:, None] == al[None, :]))
                L = np.select([null, ex, ov], [-1, 2, 1], 0)
            else:
                L = np.select([null, ov & (w <= NARROW[a]), ov], [-1, 2, 1], 0)
            L = L.astype(np.int8)
        elif a == "occ":
            rt, rc = R["occ_title"].to_numpy(dtype=object), R["occ_cat"].to_numpy(dtype=object)
            at, ac = A["occ_title"].to_numpy(dtype=object), A["occ_cat"].to_numpy(dtype=object)
            rtn, rcn, atn, acn = _isnull(R["occ_title"]), _isnull(R["occ_cat"]), _isnull(A["occ_title"]), _isnull(A["occ_cat"])
            t_both = ~rtn[:, None] & ~atn[None, :]
            c_both = ~rcn[:, None] & ~acn[None, :]
            t_eq = t_both & (rt[:, None] == at[None, :])
            c_eq = c_both & (rc[:, None] == ac[None, :])
            L = np.select([t_eq, c_eq, t_both | c_both], [2, 1, 0], -1).astype(np.int8)
        elif a == "hobby":
            ah = [next(iter(s)) if s else None for s in _listset(A["hb"])]
            ah = np.array(ah, dtype=object)
            an = np.array([h is None for h in ah])
            for i, S in enumerate(_listset(R["hb"])):
                if S:
                    L[i] = np.where(an, -1, np.isin(ah, list(S)).astype(np.int8))
        elif a in LISTCOUNT or a == "visit":
            col = {"visit": "vis_y"}.get(a, a)
            asets = _listset(A[col])
            vocab = sorted(set().union(*asets)) if asets else []
            idx = {v: j for j, v in enumerate(vocab)}
            M = np.zeros((len(vocab), na), dtype=np.int16)
            for k, s in enumerate(asets):
                for v in s:
                    M[idx[v], k] = 1
            an = np.array([not s for s in asets])
            if a == "visit":
                ym_sets = _listset(A["vis_ym"])
            for i, S in enumerate(_listset(R[col])):
                if not S:
                    continue
                cnt = sum((M[idx[v]] for v in S if v in idx), np.zeros(na, dtype=np.int16))
                if a == "visit":
                    rym = _listset(R["vis_ym"])[i]
                    ym_hit = np.array([bool(rym & s) for s in ym_sets])
                    L[i] = np.where(an, -1, np.where(ym_hit, 2, np.where(cnt >= 1, 1, 0)))
                else:
                    L[i] = np.where(an, -1, np.where(cnt >= 2, 2, np.where(cnt >= 1, 1, 0)))
        out[a] = L
    return out


def deterministic(levels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """compatible = no attribute in disagreement; n_agree = attributes with positive agreement."""
    any_dis = np.zeros(next(iter(levels.values())).shape, dtype=bool)
    agree = np.zeros(any_dis.shape, dtype=np.int8)
    for L in levels.values():
        any_dis |= L == 0
        agree += (L >= 1).astype(np.int8)
    return ~any_dis, agree


# ------------------------------------------------------------------ Splink (DuckDB SQL), same coding
def _lvl(sql, label, null=False):
    d = {"sql_condition": sql, "label_for_charts": label}
    if null:
        d["is_null_level"] = True
    return d


def splink_comparisons(attrs) -> list[dict]:
    comps = []
    for a in attrs:
        if a == "names":
            F, P = "coalesce(nm_found_l, nm_found_r)", "coalesce(nm_l, nm_r)"
            lv = [_lvl(f"coalesce(len({F}), 0) = 0", "no residual name", True),
                  _lvl(f"list_contains({F}, {P}[1]) AND list_contains({F}, {P}[2])", "first+last"),
                  _lvl(f"list_contains({F}, {P}[2])", "last"), _lvl(f"list_contains({F}, {P}[1])", "first"),
                  _lvl("ELSE", "other names")]
        elif a in SCALAR:
            lv = [_lvl(f"{a}_l IS NULL OR {a}_r IS NULL", "null", True), _lvl(f"{a}_l = {a}_r", "equal"), _lvl("ELSE", "different")]
        elif a in ("by", "kids") or a in INTERVAL3:
            ov = f"({a}_lo_l <= {a}_hi_r AND {a}_lo_r <= {a}_hi_l)"
            w = f"greatest({a}_hi_l - {a}_lo_l, {a}_hi_r - {a}_lo_r)"
            nul = _lvl(f"{a}_lo_l IS NULL OR {a}_lo_r IS NULL", "null", True)
            if a == "by":
                lv = [nul] + [_lvl(f"{ov} AND {w} <= {t}", f"overlap w<={t}") for t in (1, 3, 6)] + \
                     [_lvl(ov, "overlap wide"), _lvl("ELSE", "disjoint")]
            elif a == "kids":
                lv = [nul, _lvl(f"kids_lo_l = kids_hi_l AND kids_lo_r = kids_hi_r AND kids_lo_l = kids_lo_r", "exact"),
                      _lvl(ov, "compatible"), _lvl("ELSE", "incompatible")]
            else:
                lv = [nul, _lvl(f"{ov} AND {w} <= {NARROW[a]}", "narrow overlap"), _lvl(ov, "overlap"), _lvl("ELSE", "disjoint")]
        elif a == "occ":
            tb = "(occ_title_l IS NOT NULL AND occ_title_r IS NOT NULL)"
            cb = "(occ_cat_l IS NOT NULL AND occ_cat_r IS NOT NULL)"
            lv = [_lvl(f"NOT ({tb} OR {cb})", "null", True), _lvl(f"{tb} AND occ_title_l = occ_title_r", "title"),
                  _lvl(f"{cb} AND occ_cat_l = occ_cat_r", "category"), _lvl("ELSE", "different")]
        elif a == "hobby":
            lv = [_lvl("coalesce(len(hb_l), 0) = 0 OR coalesce(len(hb_r), 0) = 0", "null", True),
                  _lvl("list_has_any(hb_l, hb_r)", "shared"), _lvl("ELSE", "different")]
        elif a in LISTCOUNT:
            n = f"len(list_intersect({a}_l, {a}_r))"
            lv = [_lvl(f"coalesce(len({a}_l), 0) = 0 OR coalesce(len({a}_r), 0) = 0", "null", True),
                  _lvl(f"{n} >= 2", ">=2 shared"), _lvl(f"{n} >= 1", "1 shared"), _lvl("ELSE", "none shared")]
        elif a == "visit":
            lv = [_lvl("coalesce(len(vis_y_l), 0) = 0 OR coalesce(len(vis_y_r), 0) = 0", "null", True),
                  _lvl("coalesce(list_has_any(vis_ym_l, vis_ym_r), false)", "year-month"),
                  _lvl("list_has_any(vis_y_l, vis_y_r)", "year"), _lvl("ELSE", "different")]
        comps.append({"output_column_name": a, "comparison_levels": lv})
    return comps
