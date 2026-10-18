"""Encounter-level person registry: clustering, role inference and cross-field propagation.

The physician's note is cased, so names are found there reliably; the dialogue is lowercase ASR
text. Propagating every surface form of a confirmed person to *both* fields (case-insensitively,
with ambiguity gating for names that are also common words) is what makes the anonymisation
consistent between dialogue and note and catches mentions no detector saw on its own.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .lexicon import NOT_NAMES, TITLES, Lexicon
from .spans import Span

ROLE_BASE = {"PATIENT": "PATIENT_NAME", "CLINICIAN": "CLINICIAN_NAME", "FAMILY": "RELATIVE_NAME", "PET": "PET_NAME",
             "PERSON": "PERSON"}
_SPEAKER = re.compile(r"^\[([a-z_]+)\]", re.M)


def speaker_at(text: str, pos: int) -> str | None:
    last = None
    for m in _SPEAKER.finditer(text, 0, pos + 1):
        last = m.group(1)
    return last


@dataclass
class PersonEntity:
    id: int
    tokens: set[str]
    forms: set[tuple[str, ...]] = field(default_factory=set)
    votes: Counter = field(default_factory=Counter)
    mentions: list[Span] = field(default_factory=list)
    strong: bool = False
    ambiguous: bool = False  # partial name shared by several people (never linked to one of them)
    role: str = "PERSON"
    placeholder: str = "[PERSON]"


def _name_tokens(span: Span, text: str) -> list[str]:
    toks = span.meta.get("name_tokens")
    if not toks:
        toks = [w.lower().removesuffix("'s").strip("'-") for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ][A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'\-]*", text[span.start:span.end])]
    # ASR cut-offs ("is-") and quotes must not create a second person: normalise every token source
    toks = [t.strip("'-") for t in toks]
    return [t for t in toks if t.rstrip(".") not in TITLES and t not in NOT_NAMES and len(t) >= 2]


class EntityRegistry:
    def __init__(self, lex: Lexicon, role_aware: bool = True):
        self.lex = lex
        self.role_aware = role_aware
        self.entities: list[PersonEntity] = []

    # ---------------------------------------------------------------- clustering
    def _find(self, toks: set[str]) -> PersonEntity | None:
        cands = [e for e in self.entities if toks <= e.tokens or e.tokens <= toks]
        if not cands:
            return None
        exact = [e for e in cands if e.tokens == toks]
        if len(exact) == 1:
            return exact[0]
        # a partial name ("smith") contained in two different full names (john smith / mary smith)
        # is ambiguous: never attach it to either person
        if len([e for e in cands if toks < e.tokens]) > 1:
            return None
        return max(cands, key=lambda e: (len(e.tokens & toks), len(e.mentions)))

    def add(self, span: Span, texts: dict[str, str]) -> PersonEntity | None:
        toks = _name_tokens(span, texts[span.field])
        if not toks:
            return None
        ent = self._find(set(toks))
        if ent is None:
            ambiguous = any(set(toks) < e.tokens for e in self.entities)
            ent = PersonEntity(len(self.entities), set(toks), ambiguous=ambiguous)
            self.entities.append(ent)
        ent.tokens |= set(toks)
        ent.forms.add(tuple(toks))
        ent.mentions.append(span)
        ent.strong |= bool(span.meta.get("strong")) or span.score >= 0.75 or span.field == "note"
        hint = span.meta.get("role_hint")
        if hint in {"PATIENT", "CLINICIAN", "FAMILY", "PET"}:
            ent.votes[hint] += 3
        elif hint == "TITLED":
            ent.votes["PATIENT"] += 1.5 if span.field == "note" else 1
        elif hint == "ADDRESSEE" and span.field == "dialogue":
            spk = speaker_at(texts["dialogue"], span.start)
            ent.votes["PATIENT" if spk == "doctor" else "CLINICIAN" if spk == "patient" else "PERSON"] += 1
        elif span.field == "note":
            ent.votes["PATIENT"] += 0.5
        return ent

    def merge_overlapping(self) -> None:
        """Merge entities that became connected through later mentions (john + smith -> john smith)."""
        changed = True
        while changed:
            changed = False
            for a in self.entities:
                for b in self.entities:
                    if a is not b and a.mentions and b.mentions and any(set(f) <= a.tokens for f in b.forms if len(f) > 1) \
                            and any(set(f) <= b.tokens for f in a.forms if len(f) > 1):
                        a.tokens |= b.tokens; a.forms |= b.forms; a.votes += b.votes
                        a.mentions += b.mentions; a.strong |= b.strong; b.mentions = []
                        changed = True
        self.entities = [e for e in self.entities if e.mentions]

    # ---------------------------------------------------------------- roles & placeholders
    def finalize(self) -> None:
        counts: Counter = Counter()
        for ent in self.entities:
            role, score = (ent.votes.most_common(1)[0] if ent.votes else ("PERSON", 0))
            ent.role = role if (score >= 1 and self.role_aware) else "PERSON"
            if not self.role_aware or ent.ambiguous:
                ent.role = "PERSON"
        # exactly one patient: the strongest PATIENT candidate keeps the role, others fall back to
        # their next-best role (distinct people must not share [PATIENT_NAME])
        patients = sorted((e for e in self.entities if e.role == "PATIENT"),
                          key=lambda e: (e.votes["PATIENT"], len(e.mentions)), reverse=True)
        for ent in patients[1:]:
            alt = [(r, v) for r, v in ent.votes.most_common() if r != "PATIENT" and v >= 1]
            ent.role = alt[0][0] if alt else "PERSON"
        for ent in sorted(self.entities, key=lambda e: e.mentions[0].start if e.mentions else 0):
            base = ROLE_BASE[ent.role]
            counts[base] += 1
        seen: Counter = Counter()
        for ent in sorted(self.entities, key=lambda e: (e.mentions[0].field != "note", e.mentions[0].start)):
            base = ROLE_BASE[ent.role]
            seen[base] += 1
            ent.placeholder = f"[{base}]" if (counts[base] == 1 or base == "PATIENT_NAME") else f"[{base}_{seen[base]}]"

    def entity_for(self, span: Span, texts: dict[str, str]) -> PersonEntity | None:
        toks = set(_name_tokens(span, texts[span.field]))
        return self._find(toks) if toks else None

    # ---------------------------------------------------------------- propagation
    def _token_safe(self, t: str) -> bool:
        return t not in NOT_NAMES and len(t) >= 3 and (self.lex.name_ratio(t) >= 0.3 or self.lex.zipf(t) < 3.5)

    def propagate(self, texts: dict[str, str]) -> list[Span]:
        out = []
        for ent in self.entities:
            if not ent.strong:
                continue
            for tok in sorted(ent.tokens, key=len, reverse=True):
                safe = self._token_safe(tok)
                rx = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9'])(?:(?:mr|mrs|ms|miss|dr|doctor)\.?\s+)?" + re.escape(tok) + r"(?:'s|s')?(?![A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ0-9])", re.I)
                for fld, text in texts.items():
                    cased = fld == "note" or any(c.isupper() for c in text[:2000])
                    for m in rx.finditer(text):
                        surface = m.group(0)
                        name_part = surface[surface.lower().rfind(tok):]
                        if cased and not name_part[:1].isupper():
                            continue
                        titled = surface.lower() != name_part.lower()
                        if not safe and not titled:
                            prev = re.findall(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ']+", text[max(0, m.start() - 20):m.start()].lower())[-1:]
                            nxt = re.findall(r"[a-zß-öø-ÿĀ-ɏḀ-ỿ']+", text[m.end():m.end() + 20].lower())[:1]
                            adjacent = (prev and prev[0] in ent.tokens) or (nxt and nxt[0] in ent.tokens)
                            vocative = re.match(r"\s*[,?.!]", text[m.end():m.end() + 2]) and re.search(
                                r"(?:[,.]|\b(?:hi|hello|hey|morning|thanks|bye|you))\s*$", text[max(0, m.start() - 8):m.start()], re.I)
                            if not (adjacent or vocative or (cased and fld == "note")):
                                continue
                        out.append(Span(fld, m.start(), m.end(), "PERSON", "propagation", 0.85, surface,
                                        meta={"name_tokens": [tok], "entity": ent.id},
                                        reasons=[f"same name as a confirmed person mention ({len(ent.mentions)} mention(s), "
                                                 f"role {ent.role.lower()})"]))
        return out
