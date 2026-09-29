"""Tests for analysis/product.py (spec §7)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from analysis.product import pair_compatibility, product_relation, reachable
from model.actions import Alphabet
from model.config import Config
from model.graph import Graph

AB1 = Alphabet(1)  # labels: a=0, ā=1, τ=2
A, ABAR, TAU = 0, 1, 2


def make_graph(ab: Alphabet, n_max: int, n_active: int, edges, x: int = 0) -> Graph:
    """edges: iterable of (label, u, v). Adds τ self-loops where missing."""
    Ar = np.zeros((ab.size, n_max, n_max), dtype=bool)
    for label, u, v in edges:
        Ar[label, u, v] = True
    for u in range(n_active):
        if not Ar[ab.tau, u].any():
            Ar[ab.tau, u, u] = True
    active = np.zeros(n_max, dtype=bool)
    active[:n_active] = True
    return Graph(Ar, active, x, ab)


def random_graph(ab: Alphabet, n_max: int, n0: int, q: float, seed: int) -> Graph:
    cfg = Config(m=ab.m, N_max=n_max, N0=n0, q_edge=q)
    return Graph.random(ab, cfg, np.random.default_rng(seed))


def idx(u: int, v: int, nj: int) -> int:
    return u * nj + v


# ---------------------------------------------------------------- handmade M


def test_handmade_nonparity_entry_by_entry():
    # gi: 0 -a-> 1, τ loops. gj: 0 -ā-> 1, 1 -τ-> 0, τ loop at 0 added.
    gi = make_graph(AB1, 2, 2, [(A, 0, 1)])
    gj = make_graph(AB1, 2, 2, [(ABAR, 0, 1), (TAU, 1, 0)])
    M = product_relation(gi, gj)
    assert M.shape == (4, 4) and M.dtype == bool
    expected = np.zeros((4, 4), dtype=bool)
    nj = 2
    # τ_i: self-loops at 0 and 1, v fixed
    for u in range(2):
        for v in range(2):
            expected[idx(u, v, nj), idx(u, v, nj)] = True
    # τ_j: 0->0, 1->0, u fixed
    for u in range(2):
        expected[idx(u, 0, nj), idx(u, 0, nj)] = True
        expected[idx(u, 1, nj), idx(u, 0, nj)] = True
    # sync: a at gi (0->1) with ā at gj (0->1)
    expected[idx(0, 0, nj), idx(1, 1, nj)] = True
    np.testing.assert_array_equal(M, expected)


def test_handmade_sync_needs_complement():
    # Both graphs only have label a: no sync pairing possible.
    gi = make_graph(AB1, 2, 2, [(A, 0, 1)])
    gj = make_graph(AB1, 2, 2, [(A, 0, 1)])
    M = product_relation(gi, gj)
    # Only τ moves: no state changes both coordinates.
    for s in range(4):
        for t in range(4):
            if M[s, t]:
                (u, v), (u2, v2) = divmod(s, 2), divmod(t, 2)
                assert u == u2 or v == v2


def test_handmade_j_initiated_sync_and_rectangular_sizes():
    # N_max differ: gi has 2 nodes, gj 3. gj offers a (1->2), gi answers ā (0->1).
    gi = make_graph(AB1, 2, 2, [(ABAR, 0, 1)])
    gj = make_graph(AB1, 3, 3, [(A, 1, 2)])
    M = product_relation(gi, gj)
    assert M.shape == (6, 6)
    assert M[idx(0, 1, 3), idx(1, 2, 3)]
    assert M.sum() == 6 + 6 - 6 + 1  # τ self-loops coincide (identity), plus one sync


def test_inactive_nodes_have_no_edges():
    gi = make_graph(AB1, 3, 2, [(A, 0, 1)])
    gj = make_graph(AB1, 3, 2, [(ABAR, 0, 1)])
    M = product_relation(gi, gj)
    # (2, 2) is fully inactive: no outgoing edges.
    assert not M[idx(2, 2, 3)].any()
    # No edge leads from an all-active state to one involving inactive node 2,
    # so those states are unreachable from any valid start.
    for u in range(2):
        for v in range(2):
            for t in np.flatnonzero(M[idx(u, v, 3)]):
                assert 2 not in divmod(int(t), 3)


def test_handmade_parity_entry_by_entry():
    gi = make_graph(AB1, 2, 2, [(A, 0, 1)])
    gj = make_graph(AB1, 2, 2, [(ABAR, 0, 1), (TAU, 1, 0)])
    Mp = product_relation(gi, gj, parity=True)
    n = 4
    assert Mp.shape == (8, 8)
    exp = np.zeros((8, 8), dtype=bool)
    # p=0 (i moves): τ_i self-loops, sync (0,0)->(1,1), decline pass where i has sync (u=0)
    for s in range(n):
        exp[s, n + s] = True  # τ_i self-loops cover every (u, v)
    exp[idx(0, 0, 2), n + idx(1, 1, 2)] = True
    # p=1 (j moves): τ_j: v=0->0, v=1->0; sync same; pass where j has sync (v=0)
    for u in range(2):
        exp[n + idx(u, 0, 2), idx(u, 0, 2)] = True
        exp[n + idx(u, 1, 2), idx(u, 0, 2)] = True
    exp[n + idx(0, 0, 2), idx(1, 1, 2)] = True
    np.testing.assert_array_equal(Mp, exp)


def test_parity_decline_pass_only_with_sync_label():
    # gi has no sync labels at node 1; gj has none at all.
    gi = make_graph(AB1, 2, 2, [(A, 0, 0), (TAU, 1, 0)])
    gj = make_graph(AB1, 2, 2, [])
    Mp = product_relation(gi, gj, parity=True)
    n = 4
    # From p=0 at u=0 i can decline-pass; at u=1 its only move is τ to u=0.
    assert Mp[idx(0, 1, 2), n + idx(0, 1, 2)]
    assert not Mp[idx(1, 1, 2), n + idx(1, 1, 2)]
    assert Mp[idx(1, 1, 2), n + idx(0, 1, 2)]


# ------------------------------------------------------- brute force agreement


def brute_force_successors(gi: Graph, gj: Graph, u: int, v: int) -> set[tuple[int, int]]:
    ab = gi.alphabet
    out = set()
    for u2 in np.flatnonzero(gi.A[ab.tau, u]):
        out.add((int(u2), v))
    for v2 in np.flatnonzero(gj.A[ab.tau, v]):
        out.add((u, int(v2)))
    for s in ab.sync:
        c = ab.complement(s)
        for u2 in np.flatnonzero(gi.A[s, u]):
            for v2 in np.flatnonzero(gj.A[c, v]):
                out.add((int(u2), int(v2)))
    return out


def brute_force_reachable(gi: Graph, gj: Graph) -> set[tuple[int, int]]:
    seen = {(gi.x, gj.x)}
    q = deque(seen)
    while q:
        u, v = q.popleft()
        for s in brute_force_successors(gi, gj, u, v):
            if s not in seen:
                seen.add(s)
                q.append(s)
    return seen


@pytest.mark.parametrize("seed", range(10))
def test_nonparity_matches_brute_force(seed):
    ab = Alphabet(2)
    gi = random_graph(ab, 6, 4, 0.25, seed)
    gj = random_graph(ab, 5, 3, 0.25, 1000 + seed)
    M = product_relation(gi, gj)
    nj = 5
    for u in range(6):
        for v in range(5):
            got = {divmod(int(t), nj) for t in np.flatnonzero(M[idx(u, v, nj)])}
            assert got == brute_force_successors(gi, gj, u, v)
    R = reachable(M, idx(gi.x, gj.x, nj))
    assert {divmod(int(t), nj) for t in np.flatnonzero(R)} == brute_force_reachable(gi, gj)


# ------------------------------------------------------------------ reachable


@pytest.mark.parametrize("seed", range(5))
def test_reachable_contains_start_and_is_closed(seed):
    ab = Alphabet(2)
    gi = random_graph(ab, 8, 5, 0.1, seed)
    gj = random_graph(ab, 8, 5, 0.1, 50 + seed)
    for parity in (False, True):
        M = product_relation(gi, gj, parity=parity)
        start = idx(gi.x, gj.x, 8)
        R = reachable(M, start)
        assert R.dtype == bool and R.shape == (M.shape[0],)
        assert R[start]
        assert not (M[R].any(axis=0) & ~R).any()


def test_reachable_isolated_start():
    M = np.zeros((3, 3), dtype=bool)
    M[1, 2] = True
    R = reachable(M, 0)
    np.testing.assert_array_equal(R, [True, False, False])


def test_reachable_chain():
    M = np.zeros((5, 5), dtype=bool)
    for k in range(4):
        M[k, k + 1] = True
    assert reachable(M, 2).tolist() == [False, False, True, True, True]


# --------------------------------------------------------------------- parity


@pytest.mark.parametrize("seed", range(5))
def test_parity_doubles_dimension_and_flips(seed):
    ab = Alphabet(2)
    gi = random_graph(ab, 6, 4, 0.2, seed)
    gj = random_graph(ab, 5, 4, 0.2, 77 + seed)
    M = product_relation(gi, gj)
    Mp = product_relation(gi, gj, parity=True)
    n = 30
    assert M.shape == (n, n) and Mp.shape == (2 * n, 2 * n)
    assert not Mp[:n, :n].any() and not Mp[n:, n:].any()
    # Every non-parity edge is realizable from some parity (τ_i from p=0, etc.)
    union = Mp[:n, n:] | Mp[n:, :n]
    assert not (M & ~union).any()
    # Sync block identical from both parities.
    ns = [s for s in ab.sync]
    for u in range(6):
        for v in range(5):
            s = idx(u, v, 5)
            for t in range(n):
                u2, v2 = divmod(t, 5)
                sync = any(gi.A[x, u, u2] and gj.A[ab.complement(x), v, v2] for x in ns)
                if sync:
                    assert Mp[s, n + t] and Mp[n + s, t]


# --------------------------------------------------------- pair compatibility


def test_pair_compatibility_zero_without_sync_edges():
    ab = Alphabet(2)
    gi = random_graph(ab, 6, 5, 0.4, 3)
    gj = make_graph(ab, 6, 5, [(ab.tau, u, (u + 1) % 5) for u in range(5)])
    assert pair_compatibility(gi, gj) == 0
    assert pair_compatibility(gj, gi) == 0


def test_pair_compatibility_handmade():
    gi = make_graph(AB1, 2, 2, [(A, 0, 1)])
    gj = make_graph(AB1, 2, 2, [(ABAR, 0, 1), (TAU, 1, 0)])
    # reachable from (0,0): (0,0), (1,1), (1,0); only (0,0) has an outgoing sync edge
    assert pair_compatibility(gi, gj) == 1


@pytest.mark.parametrize("seed", range(5))
def test_pair_compatibility_monotone_under_edge_addition(seed):
    ab = Alphabet(2)
    rng = np.random.default_rng(seed)
    gi = random_graph(ab, 8, 6, 0.08, seed)
    gj = random_graph(ab, 8, 6, 0.08, 200 + seed)
    prev = pair_compatibility(gi, gj)
    for _ in range(15):
        g = gi if rng.random() < 0.5 else gj
        label = int(rng.integers(ab.size))
        u, v = rng.integers(6, size=2)
        g.A[label, u, v] = True
        cur = pair_compatibility(gi, gj)
        assert cur >= prev
        prev = cur
