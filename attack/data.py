"""Step 1: load encounters and build the attacker's view (dialogue + note)."""
from __future__ import annotations

import json
import re

# Placeholders emitted by the pipeline: [PATIENT_NAME], [PERSON_2], [DATE], [LOCATION: Middle East], ...
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*(?:_\d+)?(?::[^\]]*)?\]")
NOTE_SEPARATOR = "\n\n--- CLINICAL NOTE ---\n"


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def to_record(row: dict) -> dict:
    text = (row.get("dialogue") or "") + NOTE_SEPARATOR + (row.get("note") or "")
    n_placeholders = len(PLACEHOLDER_RE.findall(text))
    return {
        "encounter_id": row["encounter_id"],
        "text": text,
        "n_placeholders": n_placeholders,
        "masking_density": 1000.0 * n_placeholders / max(len(text), 1),
        "n_icd_codes": len(row.get("icd10_codes") or []),
        "icd_descriptions": tuple(c.get("description") or "" for c in row.get("icd10_codes") or []),
        # External causes and activities (V00-Y99, e.g. "Activity, golf"): facts the codes already state.
        "icd_external": tuple(c.get("description") or "" for c in row.get("icd10_codes") or []
                              if (c.get("code") or " ")[0].upper() in "VWXY"),
    }
