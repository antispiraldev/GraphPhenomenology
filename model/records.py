"""Tokens, loop records and the append-only library (spec §3.3, §5.5–5.7).

Types and Library are implemented in G0. Functions marked `G1` are frozen
signatures to be implemented in segment G1 (worktree: records).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Literal

from model.actions import Alphabet


class TokenKind(Enum):
    SELF_OFFER = "SELF_OFFER"
    SELF_TAU = "SELF_TAU"
    SELF_ANY = "SELF_ANY"
    OTHER_OFFER = "OTHER_OFFER"
    OTHER_ANY = "OTHER_ANY"
    REF = "REF"


_SELF_KINDS = {TokenKind.SELF_OFFER, TokenKind.SELF_TAU, TokenKind.SELF_ANY}
_OTHER_KINDS = {TokenKind.OTHER_OFFER, TokenKind.OTHER_ANY}


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    label: int | None = None   # set iff kind is SELF_OFFER / OTHER_OFFER (the offered σ)
    ref: int | None = None     # set iff kind is REF (record id in the owner's library)

    def __post_init__(self) -> None:
        has_label = self.kind in (TokenKind.SELF_OFFER, TokenKind.OTHER_OFFER)
        if has_label != (self.label is not None):
            raise ValueError(f"{self.kind} label mismatch: {self.label}")
        if (self.kind is TokenKind.REF) != (self.ref is not None):
            raise ValueError(f"{self.kind} ref mismatch: {self.ref}")

    @property
    def is_self(self) -> bool:
        return self.kind in _SELF_KINDS

    @property
    def is_other(self) -> bool:
        return self.kind in _OTHER_KINDS

    @staticmethod
    def self_offer(label: int) -> Token:
        return Token(TokenKind.SELF_OFFER, label=label)

    @staticmethod
    def other_offer(label: int) -> Token:
        return Token(TokenKind.OTHER_OFFER, label=label)

    @staticmethod
    def make_ref(record_id: int) -> Token:
        return Token(TokenKind.REF, ref=record_id)

    def __repr__(self) -> str:
        if self.label is not None:
            return f"{self.kind.value}({self.label})"
        if self.ref is not None:
            return f"REF({self.ref})"
        return self.kind.value


SELF_TAU = Token(TokenKind.SELF_TAU)
SELF_ANY = Token(TokenKind.SELF_ANY)
OTHER_ANY = Token(TokenKind.OTHER_ANY)


class _SelfSentinel:
    """`other = SELF` on saturated records (§5.6)."""

    _instance: _SelfSentinel | None = None

    def __new__(cls) -> _SelfSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SELF"

    def __reduce__(self):
        return (_SelfSentinel, ())


SELF = _SelfSentinel()

Source = Literal["bootstrap", "elder"]


@dataclass(frozen=True)
class Record:
    id: int                          # index in the owner's library
    level: int                       # >= 1
    tokens: tuple[Token, ...]        # compressed word; REF only as the first token
    anchor: int                      # owner's node at the start of the recorded segment
    other: int | _SelfSentinel | None  # record id of own level-(k-1) record, SELF, or None at level 1
    partner_id: int
    saturated: bool
    source: Source


class Library:
    """Append-only record store. Record ids are library indices, and each
    appended record must be exactly one level above the current top."""

    def __init__(self) -> None:
        self._records: list[Record] = []

    def append(self, record: Record) -> None:
        if record.id != len(self._records):
            raise ValueError(f"record id {record.id} != next id {len(self._records)}")
        if record.level != self.level + 1:
            raise ValueError(f"record level {record.level} != {self.level + 1}")
        self._records.append(record)

    @property
    def next_id(self) -> int:
        return len(self._records)

    @property
    def top(self) -> Record | None:
        return self._records[-1] if self._records else None

    @property
    def level(self) -> int:
        return self._records[-1].level if self._records else 0

    def __getitem__(self, record_id: int) -> Record:
        return self._records[record_id]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)


# --- flat / extras / is_alternating / own_labels: G0. The rest: G1 (records) -

def flat(record: Record, library: Library) -> tuple[Token, ...]:
    """Recursively expand REF tokens (§3.3). Result contains no REF. (G0)"""
    out: list[Token] = []
    for tok in record.tokens:
        if tok.kind is TokenKind.REF:
            out.extend(flat(library[tok.ref], library))
        else:
            out.append(tok)
    return tuple(out)


def extras(record: Record) -> tuple[Token, ...]:
    """§5.6: all tokens at level 1; the tokens after the leading REF otherwise. (G0)"""
    if record.tokens and record.tokens[0].kind is TokenKind.REF:
        return record.tokens[1:]
    return record.tokens


def is_alternating(tokens: tuple[Token, ...]) -> bool:
    """True iff a REF-free word is non-empty, strictly alternates self/other,
    starts with a self token and ends with an other token. (G0)"""
    if not tokens or len(tokens) % 2:
        return False
    return all(t.is_self if i % 2 == 0 else t.is_other for i, t in enumerate(tokens))


def own_labels(flat_tokens: tuple[Token, ...], alphabet: Alphabet) -> list[int]:
    """Labels used by my own moves (§6): σ for SELF_OFFER(σ), complement(σ)
    for OTHER_OFFER(σ). Sorted, unique. SELF_TAU and wildcards contribute none. (G0)"""
    labels = set()
    for tok in flat_tokens:
        if tok.kind is TokenKind.SELF_OFFER:
            labels.add(tok.label)
        elif tok.kind is TokenKind.OTHER_OFFER:
            labels.add(alphabet.complement(tok.label))
    return sorted(labels)


def swap_token(token: Token) -> Token:
    """Perspective swap table (§5.7). Raises on REF."""
    raise NotImplementedError


def transmit(elder_extras: tuple[Token, ...]) -> tuple[Token, ...]:
    """(SELF_ANY,) + swap(e) + (OTHER_ANY,) (§5.7)."""
    raise NotImplementedError


def make_bootstrap_record(
    library: Library, segment_tokens: tuple[Token, ...], anchor: int, partner_id: int
) -> Record:
    """Build (not append) the next record per §5.5 and §5.6.

    segment_tokens is the encoded joint cycle (level 1) or the extra segment
    (level k+1). Level, id, REF, `other` and saturation are derived from the
    library's current top."""
    raise NotImplementedError


def make_elder_record(
    library: Library, elder_record: Record, anchor: int, partner_id: int
) -> Record:
    """Build (not append) the learner's next record by transmission (§5.7).
    elder_record must be the elder's record at level library.level + 1."""
    raise NotImplementedError
