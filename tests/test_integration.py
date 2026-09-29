"""G2 gate: episode + runner invariants (spec §11)."""

import numpy as np
import pytest

from model.actions import Alphabet
from model.agent import Agent, MindPolicy
from model.config import Config
from model.controls import MirrorPolicy, make_control_graph
from model.graph import Graph, is_monotone
from model.records import extras, flat, is_alternating
from sim.episode import run_episode
from sim.runner import build_contact, run, sample_matching


def small(**kw) -> Config:
    base = dict(epochs=15, N_agents=8)
    base.update(kw)
    return Config(**base)


def test_reproducible_trace_hash():
    assert run(small(seed=3)).trace_hash == run(small(seed=3)).trace_hash
    assert run(small(seed=3)).trace_hash != run(small(seed=4)).trace_hash


def test_isolation_single_agent_never_levels_up():
    res = run(small(N_agents=1, epochs=50))
    assert res.agents[0].level == 0
    assert all(r["levelups_bootstrap"] == 0 for r in res.rows)


def test_records_well_formed_after_run():
    res = run(small(epochs=40, seed=1))
    n_records = 0
    for a in res.agents:
        prev = None
        for r in a.library:
            n_records += 1
            assert is_alternating(flat(r, a.library))
            if prev is not None:
                assert r.level == prev.level + 1
                assert flat(r, a.library) == flat(prev, a.library) + extras(r)
            prev = r
    assert n_records > 0, "smoke config should produce some level-ups"


def test_monotone_growth_across_episodes():
    cfg = small()
    alphabet = Alphabet(cfg.m)
    rng = np.random.default_rng(0)
    a, b = (Agent(k, Graph.random(alphabet, cfg, rng)) for k in range(2))
    pol = MindPolicy(alphabet, cfg)
    for _ in range(30):
        before = (a.graph.copy(), b.graph.copy())
        levels = (a.level, b.level)
        run_episode(a, b, pol, pol, alphabet, cfg, rng)
        assert is_monotone(before[0], a.graph) and is_monotone(before[1], b.graph)
        assert a.level - levels[0] in (0, 1, 2, 3, 4) and b.level - levels[1] >= 0
    # every level-up activated a node (budget not reached here)
    assert a.graph.n_active == cfg.N0 + a.level


@pytest.mark.parametrize("kind", ["object", "mirror", "recording"])
def test_controls_never_write_records(kind):
    cfg = small(N_agents=4, controls={kind: 4}, contact="erdos_renyi", p_contact=1.0)
    res = run(cfg)
    assert all(len(a.library) == 0 for a in res.agents if a.kind != "mind")
    assert sum(r["mind_control_windows"] for r in res.rows) > 0


def test_control_test_can_fail():
    """With require_mutual=false a mind levels up against another mind, so
    the one-sided machinery used by E0 is live (§11)."""
    cfg = small(require_mutual=False, epochs=5, N_agents=2, contact="pairs")
    res = run(cfg)
    assert any(a.level > 0 for a in res.agents)


def test_mind_vs_mirror_no_one_sided():
    cfg = small(N_agents=1, controls={"mirror": 1}, contact="pairs", epochs=30)
    alphabet = Alphabet(cfg.m)
    rng = np.random.default_rng(0)
    mind = Agent(0, Graph.random(alphabet, cfg, rng))
    mirror = Agent(1, make_control_graph("mirror", alphabet, cfg, rng), kind="mirror")
    for _ in range(30):
        res = run_episode(mind, mirror, MindPolicy(alphabet, cfg), MirrorPolicy(alphabet), alphabet, cfg, rng)
        assert not any(w.one_sided for w in res.window_stats)


def test_contact_graphs_and_matching():
    rng = np.random.default_rng(0)
    for contact in ("pairs", "ring", "erdos_renyi", "small_world"):
        adj = build_contact(10, Config(contact=contact), rng)
        assert (adj == adj.T).all() and not adj.diagonal().any()
        pairs = sample_matching(adj, rng)
        ids = [x for p in pairs for x in p]
        assert len(ids) == len(set(ids))
        assert all(adj[i, j] for i, j in pairs)
    pairs_adj = build_contact(6, Config(contact="pairs"), rng)
    assert sorted(sample_matching(pairs_adj, rng)) == [(0, 1), (2, 3), (4, 5)]
