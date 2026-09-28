"""G0 tests for the shared core: graph basics and word flattening."""

import numpy as np
import pytest

from model.graph import Graph
from model.records import OTHER_ANY, SELF_ANY, SELF_TAU, Library, Record, Token, extras, flat, is_alternating


def test_random_graph_shape_and_tau(alphabet, cfg, rng):
    g = Graph.random(alphabet, cfg, rng)
    assert g.A.shape == (alphabet.size, cfg.N_max, cfg.N_max)
    assert g.n_active == cfg.N0 and g.active[: cfg.N0].all()
    assert not g.A[:, cfg.N0 :, :].any() and not g.A[:, :, cfg.N0 :].any()
    for v in range(cfg.N0):
        assert alphabet.tau in g.available(v)
    assert 0 <= g.x < cfg.N0


def test_step_and_self_loops(alphabet, cfg, rng):
    g = Graph.random(alphabet, cfg, rng)
    missing = [s for s in alphabet.sync if s not in g.available()]
    if missing:
        with pytest.raises(ValueError):
            g.step(missing[0], rng)
    g.add_self_loops(list(alphabet.sync))
    for v in range(cfg.N0):
        assert g.available(v) == list(alphabet.labels)
    assert g.available(cfg.N0) == []
    x = g.step(alphabet.tau, rng)
    assert g.active[x]


def test_copy_is_independent(alphabet, cfg, rng):
    g = Graph.random(alphabet, cfg, rng)
    h = g.copy()
    h.A[:] = True
    assert not np.array_equal(g.A, h.A)


def test_flat_extras_alternation():
    lib = Library()
    w1 = (Token.self_offer(0), Token.other_offer(4))
    lib.append(Record(0, 1, w1, 0, None, 1, False, "bootstrap"))
    w2 = (Token.make_ref(0), SELF_ANY, OTHER_ANY)
    lib.append(Record(1, 2, w2, 0, 0, 1, False, "bootstrap"))
    assert flat(lib[1], lib) == w1 + (SELF_ANY, OTHER_ANY)
    assert extras(lib[0]) == w1 and extras(lib[1]) == (SELF_ANY, OTHER_ANY)
    assert is_alternating(flat(lib[1], lib))
    assert not is_alternating(())
    assert not is_alternating((SELF_TAU,))
    assert not is_alternating((OTHER_ANY, SELF_ANY))
