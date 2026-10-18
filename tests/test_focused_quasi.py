"""Focused quasi-identifier cases (height, weight, BMI, family size, structured IDs, occupations)."""
import pytest

from conftest import make_record
from medanon.evaluation.focused_cases import CASES


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{c[1]}-{i}" for i, c in enumerate(CASES)])
def test_focused_case(anonymizer, case):
    cat, kind, dlg, note, icd, gone, kept = case
    (out, _), = anonymizer.anonymize_dataset([make_record(dlg, note, icd=icd)])[0]
    text = out["dialogue"] + out["note"]
    assert [g for g in gone if g.lower() in text.lower()] == []
    assert [k for k in kept if k not in text] == []
