"""Tests for the §4.3 mind policy (MindPolicy)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from model.actions import Alphabet
from model.agent import Agent, MindPolicy, ProbeState
from model.config import Config
from model.graph import Graph
from model.records import (
    OTHER_ANY,
    SELF_ANY,
    SELF_TAU,
    Library,
    Record,
    Token,
)
from model.trace import ProbeEvent, ProbeEventKind, Tick

AL = Alphabet(3)          # a1..a3 = 0..2, ā1..ā3 = 3..5, τ = 6
TAU = AL.tau
ME, PARTNER = 7, 8        # non-trivial ids to check agent_id on events
K = ProbeEventKind


def make_graph(available: list[int], n_max: int = 4) -> Graph:
    """Single active node 0 with self-loops for `available` (τ always added)."""
    A = np.zeros((AL.size, n_max, n_max), dtype=bool)
    for label in set(available) | {TAU}:
        A[label, 0, 0] = True
    active = np.zeros(n_max, dtype=bool)
    active[0] = True
    return Graph(A, active, 0, AL)


def make_agent(available: list[int], records: list[tuple[Token, ...]] = ()) -> Agent:
    lib = Library()
    for i, tokens in enumerate(records):
        lib.append(
            Record(
                id=i, level=i + 1, tokens=tokens, anchor=0,
                other=None if i == 0 else i - 1, partner_id=PARTNER,
                saturated=False, source="bootstrap",
            )
        )
    return Agent(id=ME, graph=make_graph(available), library=lib)


def policy(**kw) -> MindPolicy:
    return MindPolicy(AL, replace(Config(), **kw))


def my_tick(action: int, response: int | None, accepted: bool, t: int = 0) -> Tick:
    return Tick(t, ME, PARTNER, action, response, accepted, 0, 0)


def their_tick(action: int, response: int | None, accepted: bool, t: int = 1) -> Tick:
    return Tick(t, PARTNER, ME, action, response, accepted, 0, 0)


def ev(kind: ProbeEventKind, rid: int, pos: int) -> ProbeEvent:
    return ProbeEvent(ME, kind, rid, pos)


ALL = list(AL.labels)


# --- full probes -------------------------------------------------------------

def test_level1_probe_success():
    word = (Token.self_offer(0), Token.other_offer(1), SELF_TAU, OTHER_ANY)
    agent = make_agent(ALL, [word])
    pol = policy(p_probe=1.0)
    rng = np.random.default_rng(0)

    label, evs = pol.act(agent, rng)
    assert label == 0 and evs == [ev(K.START, 0, 0)]
    assert pol.observe(agent, my_tick(0, AL.complement(0), True)) == [ev(K.ADVANCE, 0, 1)]

    resp, evs = pol.respond(agent, 1, rng)
    assert resp == AL.complement(1) and evs == []
    assert pol.observe(agent, their_tick(1, resp, True)) == [ev(K.ADVANCE, 0, 2)]

    label, evs = pol.act(agent, rng)
    assert label == TAU and evs == []
    assert pol.observe(agent, my_tick(TAU, None, False)) == [ev(K.ADVANCE, 0, 3)]

    assert pol.observe(agent, their_tick(TAU, None, False)) == [ev(K.SUCCESS, 0, 4)]
    assert agent.probe is None


def test_level2_ref_probe_success():
    rec0 = (Token.self_offer(0), Token.other_offer(1))
    rec1 = (Token.make_ref(0), Token.self_offer(2), Token.other_offer(5))
    agent = make_agent(ALL, [rec0, rec1])
    pol = policy(p_probe=1.0, beta=100.0)   # effectively always the level-2 record
    rng = np.random.default_rng(1)

    label, evs = pol.act(agent, rng)
    assert label == 0 and evs == [ev(K.START, 1, 0)]
    assert agent.probe.word == rec0 + rec1[1:]
    assert pol.observe(agent, my_tick(0, 3, True)) == [ev(K.ADVANCE, 1, 1)]
    resp, _ = pol.respond(agent, 1, rng)
    assert resp == 4
    assert pol.observe(agent, their_tick(1, 4, True)) == [ev(K.ADVANCE, 1, 2)]
    label, evs = pol.act(agent, rng)
    assert label == 2 and evs == []
    assert pol.observe(agent, my_tick(2, 5, True)) == [ev(K.ADVANCE, 1, 3)]
    resp, _ = pol.respond(agent, 5, rng)
    assert resp == 2
    assert pol.observe(agent, their_tick(5, 2, True)) == [ev(K.SUCCESS, 1, 4)]
    assert agent.probe is None


# --- aborts --------------------------------------------------------------------

def test_abort_self_offer_unavailable_at_start():
    agent = make_agent([1, 2], [(Token.self_offer(0), OTHER_ANY)])
    pol = policy(p_probe=1.0)
    label, evs = pol.act(agent, np.random.default_rng(0))
    assert evs == [ev(K.START, 0, 0), ev(K.ABORT, 0, 0)]
    assert agent.probe is None
    assert label in agent.graph.available()
    # nothing more happens in observe
    assert pol.observe(agent, my_tick(label, None, False)) == []


def test_abort_self_offer_unavailable_mid_probe():
    word = (Token.self_offer(1), OTHER_ANY, Token.self_offer(0), OTHER_ANY)
    agent = make_agent([1, 2], [word])
    agent.probe = ProbeState(0, word, 2)
    label, evs = policy().act(agent, np.random.default_rng(0))
    assert evs == [ev(K.ABORT, 0, 2)]
    assert agent.probe is None and label in agent.graph.available()


def test_abort_self_offer_declined():
    agent = make_agent(ALL, [(Token.self_offer(0), OTHER_ANY)])
    pol = policy(p_probe=1.0)
    label, _ = pol.act(agent, np.random.default_rng(0))
    assert label == 0
    assert pol.observe(agent, my_tick(0, TAU, False)) == [ev(K.ABORT, 0, 0)]
    assert agent.probe is None


def test_abort_other_offer_mismatch_in_respond_no_double_abort():
    word = (SELF_TAU, Token.other_offer(1))
    agent = make_agent(ALL, [word])
    agent.probe = ProbeState(0, word, 1)
    pol = policy()
    resp, evs = pol.respond(agent, 2, np.random.default_rng(0))
    assert evs == [ev(K.ABORT, 0, 1)]
    assert agent.probe is None
    assert resp in agent.graph.available()
    assert pol.observe(agent, their_tick(2, resp, resp == AL.complement(2))) == []


def test_abort_partner_tau_when_other_offer_expected():
    word = (SELF_TAU, Token.other_offer(1))
    agent = make_agent(ALL, [word])
    agent.probe = ProbeState(0, word, 1)
    assert policy().observe(agent, their_tick(TAU, None, False)) == [ev(K.ABORT, 0, 1)]
    assert agent.probe is None


def test_abort_other_offer_matched_but_not_accepted():
    # Expected partner offer arrives, but my complement is unavailable.
    word = (SELF_TAU, Token.other_offer(1))
    agent = make_agent([0, 2], [word])
    agent.probe = ProbeState(0, word, 1)
    pol = policy()
    resp, evs = pol.respond(agent, 1, np.random.default_rng(0))
    assert evs == []
    assert pol.observe(agent, their_tick(1, resp, False)) == [ev(K.ABORT, 0, 1)]
    assert agent.probe is None


# --- wildcards -------------------------------------------------------------

def test_self_any_explores_and_advances_even_if_declined():
    word = (SELF_ANY, OTHER_ANY)
    agent = make_agent([0, 1], [word])
    pol = policy(p_probe=1.0)
    label, evs = pol.act(agent, np.random.default_rng(0))
    assert evs == [ev(K.START, 0, 0)]
    assert label in agent.graph.available()
    assert pol.observe(agent, my_tick(label, TAU, False)) == [ev(K.ADVANCE, 0, 1)]


@pytest.mark.parametrize("tick", [
    their_tick(0, TAU, False),        # declined partner offer
    their_tick(0, 3, True),           # accepted partner offer
    their_tick(TAU, None, False),     # partner τ
])
def test_other_any_advances(tick):
    word = (SELF_TAU, OTHER_ANY)
    agent = make_agent(ALL, [word])
    agent.probe = ProbeState(0, word, 1)
    pol = policy()
    if tick.action != TAU:
        _, evs = pol.respond(agent, tick.action, np.random.default_rng(0))
        assert evs == []
    assert pol.observe(agent, tick) == [ev(K.SUCCESS, 0, 2)]
    assert agent.probe is None


# --- statistics -----------------------------------------------------------

N = 20000


def test_reactivity_rate():
    avail = [0, 3, 4]              # complement of 1 is 4, available; τ added → 4 labels
    agent = make_agent(avail)
    pol = policy(rho=0.5)
    rng = np.random.default_rng(3)
    n_avail = len(agent.graph.available())
    hits = 0
    for _ in range(N):
        r, evs = pol.respond(agent, 1, rng)
        assert evs == [] and r in agent.graph.available()
        hits += r == 4
    # complement from the ρ branch, plus by chance from the uniform fallback
    expected = 0.5 + 0.5 / n_avail
    assert abs(hits / N - expected) < 0.015


def test_reactivity_never_complement_when_unavailable():
    agent = make_agent([0, 2])
    pol = policy(rho=1.0)
    rng = np.random.default_rng(4)
    for _ in range(500):
        r, _ = pol.respond(agent, 1, rng)
        assert r in agent.graph.available() and r != 4


def test_probe_start_rate():
    agent = make_agent(ALL, [(SELF_TAU, OTHER_ANY)])
    pol = policy(p_probe=0.3)
    rng = np.random.default_rng(5)
    starts = 0
    for _ in range(N):
        _, evs = pol.act(agent, rng)
        starts += any(e.kind is K.START for e in evs)
        pol.reset_episode(agent)
    assert abs(starts / N - 0.3) < 0.015


def test_record_choice_softmax_over_levels():
    rec0 = (SELF_TAU, OTHER_ANY)
    recs = [rec0, (Token.make_ref(0), SELF_TAU, OTHER_ANY), (Token.make_ref(1), SELF_TAU, OTHER_ANY)]
    agent = make_agent(ALL, recs)
    beta = 1.0
    pol = policy(p_probe=1.0, beta=beta)
    rng = np.random.default_rng(6)
    counts = Counter()
    for _ in range(N):
        label, evs = pol.act(agent, rng)
        assert label == TAU
        counts[evs[0].record_id] += 1
        pol.reset_episode(agent)
    w = np.exp(beta * np.array([1, 2, 3]))
    p = w / w.sum()
    for rid in range(3):
        assert abs(counts[rid] / N - p[rid]) < 0.015


def test_explore_uniform_over_available():
    avail = [0, 2, 5]
    agent = make_agent(avail)              # empty library → always explore
    pol = policy()
    rng = np.random.default_rng(7)
    counts = Counter(pol.act(agent, rng)[0] for _ in range(N))
    labels = agent.graph.available()
    assert set(counts) == set(labels)
    for label in labels:
        assert abs(counts[label] / N - 1 / len(labels)) < 0.015


# --- misc -------------------------------------------------------------------

def test_reset_episode_clears_probe():
    word = (SELF_TAU, OTHER_ANY)
    agent = make_agent(ALL, [word])
    agent.probe = ProbeState(0, word, 1)
    x = agent.graph.x
    policy().reset_episode(agent)
    assert agent.probe is None and agent.graph.x == x


def test_act_always_available_and_events_owned():
    rec0 = (Token.self_offer(0), Token.other_offer(1))
    rec1 = (Token.make_ref(0), Token.self_offer(5), OTHER_ANY)
    pol = policy(p_probe=0.5)
    rng = np.random.default_rng(8)
    for avail in ([0], [5], [1, 2], ALL):
        agent = make_agent(avail, [rec0, rec1])
        for t in range(300):
            label, evs = pol.act(agent, rng)
            assert label in agent.graph.available()
            ok = label in agent.graph.available() and rng.random() < 0.5 and label != TAU
            evs += pol.observe(agent, my_tick(label, None if label == TAU else 0, ok, t))
            offer = int(rng.integers(AL.size))
            if offer != TAU:
                r, e2 = pol.respond(agent, offer, rng)
                evs += e2
                acc = r == AL.complement(offer) and r in agent.graph.available()
                evs += pol.observe(agent, their_tick(offer, r, acc, t))
            else:
                evs += pol.observe(agent, their_tick(TAU, None, False, t))
            assert all(e.agent_id == ME for e in evs)


def _run(seed: int) -> list:
    rec0 = (Token.self_offer(0), Token.other_offer(1))
    rec1 = (Token.make_ref(0), SELF_ANY, OTHER_ANY)
    agent = make_agent([0, 1, 4, 5], [rec0, rec1])
    pol = policy(p_probe=0.5)
    rng = np.random.default_rng(seed)
    out = []
    for t in range(200):
        label, e1 = pol.act(agent, rng)
        e1 += pol.observe(agent, my_tick(label, None if label == TAU else 3, label == 0, t))
        r, e2 = pol.respond(agent, t % 6, rng)
        out.append((label, r, tuple(e1), tuple(e2)))
    return out


def test_deterministic_with_fixed_seed():
    assert _run(11) == _run(11)
    assert _run(11) != _run(12)
