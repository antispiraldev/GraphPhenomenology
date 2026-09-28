"""Agent graph: dense Boolean label matrices over a fixed node budget (spec §3.2, §6).

Frozen signatures; implemented in segment G1 (worktree: graph).
Dense numpy bool arrays only (N_max <= 256 per §12; sparse is out of scope).
"""

from __future__ import annotations

import numpy as np

from model.actions import Alphabet
from model.config import Config
from model.records import Token


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
        raise NotImplementedError

    @property
    def n_active(self) -> int:
        raise NotImplementedError

    def available(self, x: int | None = None) -> list[int]:
        """Sorted labels with at least one outgoing edge at x (default: self.x)."""
        raise NotImplementedError

    def step(self, label: int, rng: np.random.Generator) -> int:
        """Move to a uniformly random successor along `label`; return new x.
        Raises ValueError if the label is unavailable."""
        raise NotImplementedError

    def add_self_loops(self, labels: list[int]) -> None:
        """A[label][v, v] = True at every active node."""
        raise NotImplementedError

    def activate_node(self, anchor: int, label: int) -> int | None:
        """Activate the lowest inactive node v, add anchor → v on `label` and a
        τ self-loop at v. Return v, or None if |V| == N_max."""
        raise NotImplementedError

    def copy(self) -> Graph:
        raise NotImplementedError


def apply_growth(
    graph: Graph, flat_tokens: tuple[Token, ...], anchor: int, mode: str
) -> None:
    """§6 growth on unlock. mode 'generalize': activate one node (edge label =
    first own-move label in flat_tokens, else τ), then add self-loops for all
    own_labels at every active node. mode 'none': no-op."""
    raise NotImplementedError


def is_monotone(before: Graph, after: Graph) -> bool:
    """§11: A(before) <= A(after) elementwise and V(before) ⊆ V(after)."""
    raise NotImplementedError
