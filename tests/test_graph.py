"""Graph growth (spec §3.2, §6, §11)."""

from __future__ import annotations

import numpy as np
import pytest

from model.actions import Alphabet
from model.config import Config
from model.graph import Graph, apply_growth, is_monotone
from model.records import OTHER_ANY, SELF_ANY, SELF_TAU, Token


def small_graph(alphabet: Alphabet, n0: int = 3, n_max: int = 5, seed: int = 0) -> Graph:
    cfg = Config(m=alphabet.m, N0=n0, N_max=n_max)
    return Graph.random(alphabet, cfg, np.random.default_rng(seed))


def edges_only_between_active(g: Graph) -> bool:
    inactive = ~g.active
    return not g.A[:, inactive, :].any() and not g.A[:, :, inactive].any()


def every_active_has_tau(g: Graph) -> bool:
    idx = np.flatnonzero(g.active)
    return bool(g.A[g.alphabet.tau, idx, :].any(axis=1).all())


# --- activate_node ---------------------------------------------------------

def test_activate_adds_one_node_until_budget(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    assert g.activate_node(0, 1) == 3
    assert g.n_active == 4
    assert g.activate_node(0, 1) == 4
    assert g.n_active == 5
    before = g.copy()
    assert g.activate_node(0, 1) is None
    assert g.n_active == 5
    assert np.array_equal(g.A, before.A)


def test_activate_picks_lowest_inactive(alphabet):
    g = small_graph(alphabet, n0=3, n_max=6)
    g.active[4] = True  # a gap at 3
    g.A[alphabet.tau, 4, 4] = True
    assert g.activate_node(0, 0) == 3
    assert g.activate_node(0, 0) == 5


def test_activate_edges(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    before = g.copy()
    v = g.activate_node(2, 1)
    assert g.A[alphabet.tau, v, v]
    assert g.A[1, 2, v]
    diff = g.A & ~before.A
    assert diff.sum() == 2  # exactly anchor→v and the τ self-loop
    assert edges_only_between_active(g)
    assert every_active_has_tau(g)


def test_activate_with_tau_label(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    v = g.activate_node(0, alphabet.tau)
    assert g.A[alphabet.tau, 0, v] and g.A[alphabet.tau, v, v]


def test_activate_rejects_inactive_anchor(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    with pytest.raises(ValueError):
        g.activate_node(4, 0)


# --- apply_growth ----------------------------------------------------------

def test_generalize_self_loops_everywhere(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    # own labels: 0 (SELF_OFFER 0) and complement(1) = 4 (OTHER_OFFER 1)
    word = (Token.self_offer(0), Token.other_offer(1), SELF_TAU, OTHER_ANY)
    apply_growth(g, word, anchor=1, mode="generalize")
    assert g.n_active == 4
    v = 3
    assert g.A[0, 1, v]  # first own-move label
    assert g.A[alphabet.tau, v, v]
    for u in np.flatnonzero(g.active):
        assert g.A[0, u, u]
        assert g.A[alphabet.complement(1), u, u]
    assert edges_only_between_active(g)
    assert every_active_has_tau(g)


def test_generalize_first_own_label_from_other_offer(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    word = (SELF_ANY, Token.other_offer(2), Token.self_offer(1), OTHER_ANY)
    apply_growth(g, word, anchor=0, mode="generalize")
    assert g.A[alphabet.complement(2), 0, 3]


def test_generalize_tau_edge_without_own_label(alphabet):
    g = small_graph(alphabet, n0=3, n_max=5)
    before = g.copy()
    word = (SELF_TAU, OTHER_ANY, SELF_ANY, OTHER_ANY)
    apply_growth(g, word, anchor=2, mode="generalize")
    assert g.A[alphabet.tau, 2, 3] and g.A[alphabet.tau, 3, 3]
    assert (g.A & ~before.A).sum() == 2  # no self-loops added


def test_generalize_at_budget_still_adds_self_loops(alphabet):
    g = small_graph(alphabet, n0=4, n_max=4)
    apply_growth(g, (Token.self_offer(2), OTHER_ANY), anchor=0, mode="generalize")
    assert g.n_active == 4
    assert all(g.A[2, u, u] for u in range(4))


def test_none_is_noop(alphabet):
    g = small_graph(alphabet)
    before = g.copy()
    apply_growth(g, (Token.self_offer(0), Token.other_offer(1)), anchor=0, mode="none")
    assert np.array_equal(g.A, before.A)
    assert np.array_equal(g.active, before.active)
    assert g.x == before.x


def test_unknown_mode_raises(alphabet):
    with pytest.raises(ValueError):
        apply_growth(small_graph(alphabet), (), anchor=0, mode="bogus")


# --- is_monotone -----------------------------------------------------------

def test_monotone_after_growth(alphabet):
    g = small_graph(alphabet)
    before = g.copy()
    apply_growth(g, (Token.self_offer(1), Token.other_offer(0)), anchor=0, mode="generalize")
    assert is_monotone(before, g)
    assert is_monotone(g, g.copy())


def test_not_monotone_after_edge_removal(alphabet):
    g = small_graph(alphabet)
    after = g.copy()
    s, u, v = np.argwhere(after.A)[0]
    after.A[s, u, v] = False
    assert not is_monotone(g, after)
    assert is_monotone(after, g)


def test_not_monotone_after_deactivation(alphabet):
    g = small_graph(alphabet)
    after = g.copy()
    after.active[0] = False
    assert not is_monotone(g, after)


@pytest.mark.parametrize("seed", range(20))
def test_random_growth_always_monotone(seed):
    rng = np.random.default_rng(seed)
    alphabet = Alphabet(int(rng.integers(1, 5)))
    n_max = int(rng.integers(2, 20))
    n0 = int(rng.integers(1, n_max + 1))
    cfg = Config(m=alphabet.m, N0=n0, N_max=n_max, q_edge=float(rng.random()) * 0.3)
    g = Graph.random(alphabet, cfg, rng)
    makers = [
        lambda: Token.self_offer(int(rng.integers(2 * alphabet.m))),
        lambda: Token.other_offer(int(rng.integers(2 * alphabet.m))),
        lambda: SELF_TAU,
        lambda: SELF_ANY,
        lambda: OTHER_ANY,
    ]
    for _ in range(30):
        word = tuple(makers[rng.integers(len(makers))]() for _ in range(rng.integers(0, 7)))
        anchor = int(rng.choice(np.flatnonzero(g.active)))
        n_before = g.n_active
        before = g.copy()
        apply_growth(g, word, anchor, "generalize")
        assert is_monotone(before, g)
        assert g.n_active == min(n_before + 1, n_max)
        assert edges_only_between_active(g)
        assert every_active_has_tau(g)
