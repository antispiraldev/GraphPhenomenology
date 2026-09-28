"""Tests for loop records (spec §3.3, §5.5–5.7, §11)."""

from __future__ import annotations

import dataclasses

import pytest

from model.records import (
    OTHER_ANY,
    SELF,
    SELF_ANY,
    SELF_TAU,
    Library,
    Record,
    Token,
    TokenKind,
    extras,
    flat,
    is_alternating,
    make_bootstrap_record,
    make_elder_record,
    swap_token,
    transmit,
)

SO, OO = Token.self_offer, Token.other_offer


def seg(*tokens: Token) -> tuple[Token, ...]:
    return tuple(tokens)


def check_composition(lib: Library) -> None:
    """§11 composition and alternation for every record in a library."""
    for rec in lib:
        assert is_alternating(flat(rec, lib))
        assert is_alternating(extras(rec))
        if rec.level == 1:
            assert rec.other is None
            assert flat(rec, lib) == rec.tokens
        else:
            prev = lib[rec.id - 1]
            assert rec.tokens[0] == Token.make_ref(prev.id)
            assert flat(rec, lib) == flat(prev, lib) + extras(rec)
        assert all(t.kind is not TokenKind.REF for t in rec.tokens[1:])


# --- swap table ------------------------------------------------------------

@pytest.mark.parametrize(
    "src, dst",
    [
        (SO(2), OO(2)),
        (SELF_TAU, OTHER_ANY),
        (SELF_ANY, OTHER_ANY),
        (OO(4), SO(4)),
        (OTHER_ANY, SELF_ANY),
    ],
)
def test_swap_table(src, dst):
    assert swap_token(src) == dst


@pytest.mark.parametrize("tok", [SO(0), SO(5), OO(1), OO(3)])
def test_swap_is_involution_on_offers(tok):
    assert swap_token(swap_token(tok)) == tok


def test_swap_flips_side():
    for tok in (SO(0), SELF_TAU, SELF_ANY):
        assert swap_token(tok).is_other
    for tok in (OO(0), OTHER_ANY):
        assert swap_token(tok).is_self


def test_swap_rejects_ref():
    with pytest.raises(ValueError):
        swap_token(Token.make_ref(0))


# --- transmit --------------------------------------------------------------

@pytest.mark.parametrize(
    "e",
    [
        seg(SO(0), OO(1)),
        seg(SELF_TAU, OTHER_ANY),
        seg(SELF_ANY, OO(2), SO(1), OTHER_ANY),
        seg(SO(0), OO(3), SELF_TAU, OO(4), SELF_ANY, OTHER_ANY),
    ],
)
def test_transmit_alternates_and_pads(e):
    out = transmit(e)
    assert len(out) == len(e) + 2
    assert is_alternating(out)
    assert out[0] == SELF_ANY and out[-1] == OTHER_ANY
    assert out[1:-1] == tuple(swap_token(t) for t in e)


def test_transmit_rejects_non_alternating():
    with pytest.raises(ValueError):
        transmit(seg(OO(0), SO(1)))
    with pytest.raises(ValueError):
        transmit(())


# --- bootstrap -------------------------------------------------------------

SEG1 = seg(SO(0), OO(1))
SEG2 = seg(SELF_TAU, OO(2))
SEG3 = seg(SELF_ANY, OTHER_ANY, SO(1), OO(0))


def build_library(*segments: tuple[Token, ...]) -> Library:
    lib = Library()
    for i, s in enumerate(segments):
        lib.append(make_bootstrap_record(lib, s, anchor=10 + i, partner_id=7))
    return lib


def test_bootstrap_level1():
    lib = Library()
    rec = make_bootstrap_record(lib, SEG1, anchor=3, partner_id=9)
    assert len(lib) == 0  # build, not append
    assert rec.id == 0 and rec.level == 1
    assert rec.tokens == SEG1 and rec.other is None
    assert rec.anchor == 3 and rec.partner_id == 9
    assert not rec.saturated and rec.source == "bootstrap"


def test_bootstrap_levels_1_to_3_compose():
    lib = build_library(SEG1, SEG2, SEG3)
    assert [r.level for r in lib] == [1, 2, 3]
    assert [r.id for r in lib] == [0, 1, 2]
    assert lib[1].tokens == (Token.make_ref(0),) + SEG2
    assert lib[2].tokens == (Token.make_ref(1),) + SEG3
    assert lib[1].other == 0 and lib[2].other == 1
    assert not any(r.saturated for r in lib)
    assert [r.anchor for r in lib] == [10, 11, 12]
    check_composition(lib)
    assert flat(lib[2], lib) == SEG1 + SEG2 + SEG3


def test_saturation_detected():
    lib = build_library(SEG1, SEG2)
    rec = make_bootstrap_record(lib, SEG2, anchor=0, partner_id=7)
    assert rec.saturated and rec.other is SELF
    assert rec.tokens == (Token.make_ref(1),) + SEG2
    lib.append(rec)
    check_composition(lib)


def test_saturation_at_level2():
    lib = build_library(SEG1)
    rec = make_bootstrap_record(lib, SEG1, anchor=0, partner_id=7)
    assert rec.level == 2 and rec.saturated and rec.other is SELF


def test_no_saturation_against_older_level():
    # Repeating level 1's extras at level 3 is not saturation (only top counts).
    lib = build_library(SEG1, SEG2)
    rec = make_bootstrap_record(lib, SEG1, anchor=0, partner_id=7)
    assert not rec.saturated and rec.other == 1


@pytest.mark.parametrize(
    "bad",
    [
        (),
        seg(SO(0)),
        seg(OO(0), SO(1)),
        seg(SO(0), SO(1)),
        seg(SO(0), OO(1), SO(2)),
        seg(Token.make_ref(0), SO(0), OO(1)),
    ],
)
def test_bootstrap_rejects_non_alternating(bad):
    lib = build_library(SEG1)
    with pytest.raises(ValueError):
        make_bootstrap_record(lib, bad, anchor=0, partner_id=7)


# --- elder transmission ----------------------------------------------------

def test_elder_level1():
    elder_lib = build_library(SEG1)
    learner = Library()
    rec = make_elder_record(learner, elder_lib[0], anchor=5, partner_id=2)
    assert rec.level == 1 and rec.id == 0 and rec.other is None
    assert rec.tokens == transmit(SEG1)
    assert rec.source == "elder" and rec.partner_id == 2 and rec.anchor == 5
    learner.append(rec)
    check_composition(learner)


def test_elder_higher_levels_ref_own_library():
    elder_lib = build_library(SEG1, SEG2, SEG3)
    # Learner has a different level-1 record, of different length, so a REF
    # into the elder's library would produce a different flat word.
    learner = build_library(seg(SELF_TAU, OTHER_ANY, SO(2), OO(2)))
    for level in (2, 3):
        rec = make_elder_record(learner, elder_lib[level - 1], anchor=0, partner_id=1)
        assert rec.level == level and rec.source == "elder"
        ref = rec.tokens[0]
        assert ref.kind is TokenKind.REF and ref.ref == learner.top.id
        assert learner[ref.ref] is learner.top
        assert extras(rec) == transmit(extras(elder_lib[level - 1]))
        assert rec.other == learner.top.id
        learner.append(rec)
    check_composition(learner)


def test_elder_saturation():
    # Elder's level 3 repeats its level 2, so the learner's transmissions repeat too.
    elder_lib = build_library(SEG1, SEG2, SEG2)
    learner = build_library(SEG3)
    learner.append(make_elder_record(learner, elder_lib[1], anchor=0, partner_id=1))
    rec = make_elder_record(learner, elder_lib[2], anchor=0, partner_id=1)
    assert rec.saturated and rec.other is SELF
    learner.append(rec)
    check_composition(learner)


@pytest.mark.parametrize("elder_level", [1, 3])
def test_elder_wrong_level_rejected(elder_level):
    elder_lib = build_library(SEG1, SEG2, SEG3)
    learner = build_library(SEG1)  # level 1: needs an elder level-2 record
    with pytest.raises(ValueError):
        make_elder_record(learner, elder_lib[elder_level - 1], anchor=0, partner_id=1)


# --- immutability ----------------------------------------------------------

def test_records_immutable():
    lib = build_library(SEG1, SEG2)
    rec = lib[1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.level = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.tokens[1].label = 3
    assert isinstance(rec.tokens, tuple)
    assert isinstance(rec, Record)


def test_building_does_not_mutate_library():
    lib = build_library(SEG1)
    before = list(lib)
    make_bootstrap_record(lib, SEG2, anchor=0, partner_id=1)
    make_elder_record(lib, build_library(SEG1, SEG2)[1], anchor=0, partner_id=1)
    assert list(lib) == before
