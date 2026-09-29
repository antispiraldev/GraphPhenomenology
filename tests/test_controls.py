"""Tests for control agents (spec §8)."""

from __future__ import annotations

import numpy as np
import pytest

from model.actions import Alphabet
from model.agent import Agent
from model.config import Config
from model.controls import (
    MirrorPolicy,
    ObjectPolicy,
    RecordingPolicy,
    capture_script,
    make_control_graph,
)
from model.graph import Graph
from model.trace import ProbeEvent
from tests.helpers import accepted_offer, build_trace, declined_offer, tau


def graph_with(alphabet: Alphabet, labels: list[int], n_max: int = 4) -> Graph:
    """One active node (0) with a self-loop for each given label plus τ."""
    A = np.zeros((alphabet.size, n_max, n_max), dtype=bool)
    for label in set(labels) | {alphabet.tau}:
        A[label, 0, 0] = True
    active = np.zeros(n_max, dtype=bool)
    active[0] = True
    return Graph(A, active, 0, alphabet)


def make_policy(kind: str, alphabet: Alphabet):
    if kind == "object":
        return ObjectPolicy(alphabet)
    if kind == "mirror":
        return MirrorPolicy(alphabet)
    return RecordingPolicy(alphabet, (0, 4, 6, 2))


KINDS = ["object", "mirror", "recording"]


@pytest.mark.parametrize("kind", KINDS)
def test_protocol_return_types(kind, alphabet, cfg, rng):
    policy = make_policy(kind, alphabet)
    agent = Agent(0, make_control_graph(kind, alphabet, cfg, rng), kind=kind)
    trace = build_trace([accepted_offer(alphabet, 0), tau(alphabet)])
    for _ in range(50):
        label, events = policy.act(agent, rng)
        assert isinstance(label, int) and label in alphabet.labels
        assert label in agent.graph.available()
        assert events == []
        for offer in alphabet.sync:
            resp, events = policy.respond(agent, offer, rng)
            assert isinstance(resp, int) and resp in alphabet.labels
            assert events == []
        for tick in trace:
            assert policy.observe(agent, tick) == []
        assert policy.reset_episode(agent) is None
    assert agent.library.level == 0 and agent.probe is None


@pytest.mark.parametrize("kind", ["object", "mirror"])
def test_object_and_mirror_never_initiate(kind, alphabet, cfg):
    rng = np.random.default_rng(1)
    policy = make_policy(kind, alphabet)
    agent = Agent(0, make_control_graph(kind, alphabet, cfg, rng), kind=kind)
    for _ in range(200):
        label, _ = policy.act(agent, rng)
        assert label == alphabet.tau
        agent.graph.step(label, rng)


def test_mirror_complement_always_available(alphabet, cfg):
    rng = np.random.default_rng(2)
    g = make_control_graph("mirror", alphabet, cfg, rng)
    policy = MirrorPolicy(alphabet)
    agent = Agent(0, g, kind="mirror")
    for v in np.flatnonzero(g.active):
        assert g.available(int(v)) == list(alphabet.labels)
        for label in alphabet.labels:
            assert g.A[label, v, v]
    for _ in range(200):
        offer = int(rng.integers(2 * alphabet.m))
        resp, _ = policy.respond(agent, offer, rng)
        assert resp == alphabet.complement(offer)
        assert resp in g.available()
        g.step(resp, rng)


def test_object_and_recording_graph_is_plain_random(alphabet, cfg):
    g1 = make_control_graph("object", alphabet, cfg, np.random.default_rng(5))
    g2 = Graph.random(alphabet, cfg, np.random.default_rng(5))
    assert np.array_equal(g1.A, g2.A) and g1.x == g2.x
    g3 = make_control_graph("recording", alphabet, cfg, np.random.default_rng(5))
    assert np.array_equal(g3.A, g2.A)
    with pytest.raises(ValueError):
        make_control_graph("mind", alphabet, cfg, np.random.default_rng(5))


def test_object_responses_uniform_over_available(alphabet):
    rng = np.random.default_rng(3)
    available = [0, 2, 4]  # plus τ
    agent = Agent(0, graph_with(alphabet, available), kind="object")
    policy = ObjectPolicy(alphabet)
    n = 20000
    counts = np.zeros(alphabet.size, dtype=int)
    for _ in range(n):
        resp, _ = policy.respond(agent, int(rng.integers(2 * alphabet.m)), rng)
        counts[resp] += 1
    labels = agent.graph.available()
    assert set(np.flatnonzero(counts)) == set(labels)
    expected = n / len(labels)
    # chi-square, df = 3; 16.27 is the 0.001 critical value
    chi2 = sum((counts[lab] - expected) ** 2 / expected for lab in labels)
    assert chi2 < 16.27


def test_recording_script_order_and_loop(alphabet, rng):
    script = (0, 4, 1, 3)
    agent = Agent(0, graph_with(alphabet, list(alphabet.sync)), kind="recording")
    policy = RecordingPolicy(alphabet, script)
    out = []
    for i in range(10):
        if i % 2 == 0:
            label, _ = policy.act(agent, rng)
        else:
            label, _ = policy.respond(agent, 2, rng)
        out.append(label)
    assert out == [script[i % 4] for i in range(10)]


def test_recording_tau_fallback_and_decline(alphabet, rng):
    agent = Agent(0, graph_with(alphabet, [0]), kind="recording")
    policy = RecordingPolicy(alphabet, (1, 1, 0))
    assert policy.act(agent, rng)[0] == alphabet.tau     # 1 unavailable → τ
    assert policy.respond(agent, 4, rng)[0] == 1          # decline, label kept
    assert policy.act(agent, rng)[0] == 0                 # script still advanced
    assert policy.pos == 0


def test_recording_observe_does_not_advance(alphabet, rng):
    agent = Agent(0, graph_with(alphabet, list(alphabet.sync)), kind="recording")
    policy = RecordingPolicy(alphabet, (0, 1, 2))
    policy.act(agent, rng)
    for tick in build_trace([accepted_offer(alphabet, 0)] * 5):
        policy.observe(agent, tick)
    assert policy.pos == 1
    assert policy.act(agent, rng)[0] == 1


def test_recording_position_persists_across_reset(alphabet, rng):
    agent = Agent(0, graph_with(alphabet, list(alphabet.sync)), kind="recording")
    policy = RecordingPolicy(alphabet, (0, 1, 2, 3))
    policy.act(agent, rng)
    policy.respond(agent, 5, rng)
    policy.reset_episode(agent)
    assert policy.pos == 2
    assert policy.act(agent, rng)[0] == 2


def test_capture_script_hand_built(alphabet):
    t = alphabet.tau
    moves = [
        accepted_offer(alphabet, 0),   # agent 0 initiates 0
        accepted_offer(alphabet, 1),   # agent 1 offers 1; agent 0 responds 4
        tau(alphabet),                 # agent 0 τ → recorded
        tau(alphabet),                 # agent 1 τ → skipped for agent 0
        declined_offer(alphabet, 2),   # agent 0 offers 2
        declined_offer(alphabet, 5),   # agent 1 offers 5; agent 0 declines with τ
    ]
    trace = build_trace(moves, a=0, b=1, first=0)
    assert capture_script(trace, 0) == (0, 4, t, 2, t)
    # agent 1: responds 3 (to 0), offers 1, skip, τ, responds τ, offers 5
    assert capture_script(trace, 1) == (3, 1, t, t, 5)
    assert capture_script([], 0) == ()


def test_capture_script_roundtrip_through_recording(alphabet, rng):
    trace = build_trace(
        [accepted_offer(alphabet, 0), tau(alphabet), accepted_offer(alphabet, 2)],
        first=0,
    )
    script = capture_script(trace, 0)
    assert script == (0, 2)  # partner τ at t=1 skipped
    policy = RecordingPolicy(alphabet, script)
    agent = Agent(0, graph_with(alphabet, list(alphabet.sync)), kind="recording")
    assert [policy.respond(agent, 0, rng)[0] for _ in range(3)] == [0, 2, 0]


@pytest.mark.parametrize("kind", KINDS)
def test_determinism_with_fixed_seed(kind, alphabet, cfg):
    def run(seed: int):
        rng = np.random.default_rng(seed)
        g = make_control_graph(kind, alphabet, cfg, rng)
        policy = make_policy(kind, alphabet)
        agent = Agent(0, g, kind=kind)
        out = []
        for _ in range(100):
            label, _ = policy.act(agent, rng)
            resp, _ = policy.respond(agent, int(rng.integers(2 * alphabet.m)), rng)
            out.append((label, resp, g.x))
            g.step(label, rng)
        return g.A.copy(), out

    A1, o1 = run(42)
    A2, o2 = run(42)
    assert np.array_equal(A1, A2) and o1 == o2


def test_no_probe_events_type(alphabet, rng):
    # ProbeEvent imported to make the "no probe events" contract explicit.
    agent = Agent(0, graph_with(alphabet, [0]), kind="object")
    for policy in (ObjectPolicy(alphabet), MirrorPolicy(alphabet)):
        _, ev = policy.respond(agent, 0, rng)
        assert not any(isinstance(e, ProbeEvent) for e in ev)


def test_mirror_config_n0_variants(alphabet):
    cfg = Config(N0=5, N_max=8)
    g = make_control_graph("mirror", alphabet, cfg, np.random.default_rng(0))
    # self-loops only on active nodes
    assert not g.A[:, 5:, :].any() and not g.A[:, :, 5:].any()
    assert all(g.A[:, v, v].all() for v in range(5))
