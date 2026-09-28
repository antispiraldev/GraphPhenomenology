"""Static pair-compatibility analysis via Boolean Kronecker products (spec §7).

Frozen signatures; implemented in segment G1 (worktree: product).
Never called from the simulation loop.

Product state (u, v) has flat index u * N_max_j + v. With parity, index
p * (N_max_i * N_max_j) + u * N_max_j + v, where p = 0 means it's i's turn.
"""

from __future__ import annotations

import numpy as np

from model.graph import Graph


def product_relation(gi: Graph, gj: Graph, parity: bool = False) -> np.ndarray:
    """M = A_i[τ]⊗I + I⊗A_j[τ] + Σ_σ A_i[σ]⊗A_j[complement(σ)] over (OR, AND).
    With parity, τ moves and sync moves initiated by the side to move flip p.
    Returns a square bool array."""
    raise NotImplementedError


def reachable(M: np.ndarray, start: int) -> np.ndarray:
    """Bool vector of states reachable from `start` (including start)."""
    raise NotImplementedError


def pair_compatibility(gi: Graph, gj: Graph) -> int:
    """Number of product states reachable from (gi.x, gj.x) that have at
    least one outgoing sync edge (non-parity relation)."""
    raise NotImplementedError
