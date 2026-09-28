"""Agent graph: dense Boolean label matrices over a fixed node budget (spec §3.2, §6).

Graph basics are implemented in G0 because every G1 module depends on them.
Growth (activate_node, apply_growth, is_monotone) is implemented in G1
(worktree: graph).
Dense numpy bool arrays only (N_max <= 256 per §12; sparse is out of scope).
"""

from __future__ import annotations

import numpy as np

from model.actions import Alphabet
from model.config import Config
from model.records import Token, TokenKind, own_labels


class Graph:
    """A[σ][u, v] = True means an edge u → v labeled σ.

    Attributes:
      A:      bool array, shape (alphabet.size, N_max, N_max)
      active: bool array, shape (N_max,)
      x:      current node (always active)
    Invariant: edges exist only between active nodes; every active node has
    at least one outgoing τ edge.
    """

    def __init__(self, A: np.ndarray, active: np.ndarray, x: int, alphabet: Alphabet) -> None:
        self.A = A
        self.active = active
        self.x = x
        self.alphabet = alphabet

    @classmethod
    def random(cls, alphabet: Alphabet, cfg: Config, rng: np.random.Generator) -> Graph:
        """§3.2 initialization: nodes 0..N0-1 active, each ordered pair (u, v)
        of active nodes (including u == v) gets an edge per label with
        probability q_edge; add a τ self-loop where a node lacks any τ edge;
        x chosen uniformly among active nodes."""
        A = np.zeros((alphabet.size, cfg.N_max, cfg.N_max), dtype=bool)
        n = cfg.N0
        A[:, :n, :n] = rng.random((alphabet.size, n, n)) < cfg.q_edge
        no_tau = ~A[alphabet.tau, :n, :n].any(axis=1)
        idx = np.flatnonzero(no_tau)
        A[alphabet.tau, idx, idx] = True
        active = np.zeros(cfg.N_max, dtype=bool)
        active[:n] = True
        return cls(A, active, int(rng.integers(n)), alphabet)

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def available(self, x: int | None = None) -> list[int]:
        """Sorted labels with at least one outgoing edge at x (default: self.x)."""
        x = self.x if x is None else x
        return np.flatnonzero(self.A[:, x, :].any(axis=1)).tolist()

    def step(self, label: int, rng: np.random.Generator) -> int:
        """Move to a uniformly random successor along `label`; return new x.
        Raises ValueError if the label is unavailable."""
        succ = np.flatnonzero(self.A[label, self.x])
        if succ.size == 0:
            raise ValueError(f"label {label} unavailable at node {self.x}")
        self.x = int(succ[rng.integers(succ.size)])
        return self.x

    def add_self_loops(self, labels: list[int]) -> None:
        """A[label][v, v] = True at every active node."""
        idx = np.flatnonzero(self.active)
        for label in labels:
            self.A[label, idx, idx] = True

    def activate_node(self, anchor: int, label: int) -> int | None:
        """Activate the lowest inactive node v, add anchor → v on `label` and a
        τ self-loop at v. Return v, or None if |V| == N_max."""
        if not self.active[anchor]:
            raise ValueError(f"anchor {anchor} is not active")
        inactive = np.flatnonzero(~self.active)
        if inactive.size == 0:
            return None
        v = int(inactive[0])
        self.active[v] = True
        self.A[label, anchor, v] = True
        self.A[self.alphabet.tau, v, v] = True
        return v

    def copy(self) -> Graph:
        return Graph(self.A.copy(), self.active.copy(), self.x, self.alphabet)


def apply_growth(
    graph: Graph, flat_tokens: tuple[Token, ...], anchor: int, mode: str
) -> None:
    """§6 growth on unlock. mode 'generalize': activate one node (edge label =
    first own-move label in flat_tokens, else τ), then add self-loops for all
    own_labels at every active node. mode 'none': no-op."""
    if mode == "none":
        return
    if mode != "generalize":
        raise ValueError(f"unknown growth mode {mode!r}")
    alphabet = graph.alphabet
    edge_label = alphabet.tau
    for tok in flat_tokens:
        if tok.kind is TokenKind.SELF_OFFER:
            edge_label = tok.label
            break
        if tok.kind is TokenKind.OTHER_OFFER:
            edge_label = alphabet.complement(tok.label)
            break
    graph.activate_node(anchor, edge_label)
    graph.add_self_loops(own_labels(flat_tokens, alphabet))


def is_monotone(before: Graph, after: Graph) -> bool:
    """§11: A(before) <= A(after) elementwise and V(before) ⊆ V(after)."""
    if before.A.shape != after.A.shape or before.active.shape != after.active.shape:
        return False
    return bool(np.all(after.A >= before.A) and np.all(after.active >= before.active))
