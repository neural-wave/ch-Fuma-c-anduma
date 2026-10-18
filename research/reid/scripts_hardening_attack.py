"""Frozen attacker as a secondary diagnostic: baseline age_bands vs candidate cand_age_bands on dev worlds.
Attacker extraction/scoring/threshold untouched (hashes in results/frozen_attacker.sha256); only the pair list grows.

    python scripts_hardening_attack.py <runs dir> <out.json>
"""
import sys
from types import SimpleNamespace

from reid import bench

runs, out = sys.argv[1], sys.argv[2]
bench.cmd_attack(SimpleNamespace(runs=runs, worlds=["dev1", "dev2"], policies=["cand_age_bands"], threats=["A", "B", "C"]))
bench.PAIRS = (("age_bands", "cand_age_bands"),)
bench.cmd_evaluate(SimpleNamespace(runs=runs, worlds=["dev1", "dev2"], out=out, pooled=True, shuffle_policy="none"))
