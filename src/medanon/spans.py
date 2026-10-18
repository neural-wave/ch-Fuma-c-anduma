"""Detected span representation and overlap resolution."""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical PHI / quasi-identifier categories and their risk class (used in explanations).
RISK_CLASS = {
    "PERSON": "direct identifier",
    "DOB": "direct identifier",
    "ID": "direct identifier",
    "CONTACT": "direct identifier",
    "ADDRESS": "direct identifier",
    "DATE": "temporal quasi-identifier",
    "YEAR": "temporal quasi-identifier",
    "MONTH": "temporal quasi-identifier",
    "WEEKDAY": "temporal quasi-identifier",
    "HOLIDAY": "temporal quasi-identifier",
    "AGE": "demographic quasi-identifier",
    "LOCATION": "geographic quasi-identifier",
    "ORGANIZATION": "contextual quasi-identifier",
    "OCCUPATION": "contextual quasi-identifier",
    "NATIONALITY": "demographic quasi-identifier",
    "ETHNICITY": "demographic quasi-identifier (clinically relevant)",
    "LAB_VALUE": "linkable clinical measurement",
    "ANTHROPOMETRIC": "linkable body measurement (quasi-identifier)",
    "FAMILY_SIZE": "demographic quasi-identifier",
    "RELATIVE_DATE": "relative temporal expression (weak quasi-identifier)",
}
# Higher wins when overlapping spans disagree on the label.
PRIORITY = {
    "DOB": 100, "ID": 95, "CONTACT": 95, "ADDRESS": 90, "DATE": 85, "PERSON": 80, "AGE": 75,
    "ORGANIZATION": 60, "LOCATION": 55, "OCCUPATION": 50, "ETHNICITY": 46, "NATIONALITY": 45, "HOLIDAY": 40,
    "YEAR": 35, "MONTH": 30, "WEEKDAY": 25, "FAMILY_SIZE": 20, "ANTHROPOMETRIC": 12, "LAB_VALUE": 10, "RELATIVE_DATE": 5,
}


@dataclass
class Span:
    field: str
    start: int
    end: int
    label: str
    source: str
    score: float = 0.8
    text: str = ""
    meta: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def overlaps(self, other: "Span") -> bool:
        return self.field == other.field and self.start < other.end and other.start < self.end

    @property
    def sources(self) -> list[str]:
        return self.meta.setdefault("sources", [self.source])


def resolve_overlaps(spans: list[Span], protected=None) -> list[Span]:
    """Merge overlapping detections per field into non-overlapping spans.

    Within a group of mutually overlapping spans, the winner is chosen by label priority,
    then length, then score; its extent is widened to the union of the group so that no
    fragment of a detected identifier survives. Evidence of all members is kept.

    `protected(field, start, end) -> bool` flags character ranges holding clinical content: a
    member with a *different* label may only widen the winner if its extra part is not protected.
    """
    if protected is not None:
        spans = _trim_protected_members(spans, protected)
    return _resolve(spans)


def _trim_protected_members(spans: list[Span], protected) -> list[Span]:
    """Cut a span back to its overlap with a stronger, differently labelled span when the part it
    would add contains protected clinical content."""
    out = []
    for s in spans:
        lo, hi = s.start, s.end
        for o in spans:
            if o is s or o.field != s.field or o.label == s.label or not (o.start < s.end and s.start < o.end):
                continue
            if (PRIORITY.get(o.label, 0), o.end - o.start) <= (PRIORITY.get(s.label, 0), s.end - s.start):
                continue  # only the weaker span of the pair is trimmed
            if lo < o.start and protected(s.field, lo, o.start):
                lo = o.start
            if hi > o.end and protected(s.field, o.end, hi):
                hi = o.end
        if (lo, hi) != (s.start, s.end):
            s = Span(s.field, lo, hi, s.label, s.source, s.score, s.text, dict(s.meta),
                     s.reasons + ["trimmed: overlapping part with clinical content left untouched"])
        out.append(s)
    return out


def _resolve(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    by_field: dict[str, list[Span]] = {}
    for s in spans:
        by_field.setdefault(s.field, []).append(s)
    for fld, items in by_field.items():
        items.sort(key=lambda s: (s.start, -s.end))
        group: list[Span] = []
        group_end = -1
        for s in items + [None]:  # sentinel flushes the last group
            if s is not None and group and s.start < group_end:
                group.append(s)
                group_end = max(group_end, s.end)
                continue
            if group:
                # syntactic rule evidence ("X Clinic", "Dr. X") decides the label over statistical guesses
                best = max(group, key=lambda g: (bool(g.meta.get("strong")) and g.source.startswith("rule"),
                                                 PRIORITY.get(g.label, 0), g.end - g.start, g.score))
                merged = Span(fld, min(g.start for g in group), max(g.end for g in group), best.label,
                              best.source, max(g.score for g in group if g.label == best.label),
                              meta=dict(best.meta), reasons=list(best.reasons))
                srcs: list[str] = []
                for g in group:
                    for src in g.sources:
                        if src not in srcs:
                            srcs.append(src)
                    for r in g.reasons:
                        if r not in merged.reasons:
                            merged.reasons.append(r)
                merged.meta["sources"] = srcs
                for g in group:  # keep attributes only another same-label member knows (e.g. occupation category)
                    if g.label == best.label:
                        for k, v in g.meta.items():
                            merged.meta.setdefault(k, v)
                if len({g.label for g in group}) > 1:
                    merged.meta["overlapping_labels"] = sorted({g.label for g in group})
                out.append(merged)
            group = [s] if s is not None else []
            group_end = s.end if s is not None else -1
    return out
