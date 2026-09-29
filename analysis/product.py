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
    ab = gi.alphabet
    ni, nj = gi.A.shape[1], gj.A.shape[1]
    Ii = np.eye(ni, dtype=np.uint8)
    Ij = np.eye(nj, dtype=np.uint8)
    tau_i = _kron(gi.A[ab.tau], Ij)
    tau_j = _kron(Ii, gj.A[ab.tau])
    sync = _sync_relation(gi, gj)
    if not parity:
        return tau_i | tau_j | sync
    # Declined offer: the side to move has some sync label at its node;
    # neither agent moves, only the turn passes.
    has_sync_i = gi.A[list(ab.sync)].any(axis=(0, 2))
    has_sync_j = gj.A[list(ab.sync)].any(axis=(0, 2))
    pass_i = np.diag(np.repeat(has_sync_i, nj))
    pass_j = np.diag(np.tile(has_sync_j, ni))
    n = ni * nj
    M = np.zeros((2 * n, 2 * n), dtype=bool)
    M[:n, n:] = tau_i | sync | pass_i
    M[n:, :n] = tau_j | sync | pass_j
    return M


def _kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a.astype(np.uint8), b.astype(np.uint8)) > 0


def _sync_relation(gi: Graph, gj: Graph) -> np.ndarray:
    """Σ_σ A_i[σ] ⊗ A_j[complement(σ)] over all sync σ (both initiators)."""
    ab = gi.alphabet
    n = gi.A.shape[1] * gj.A.shape[1]
    S = np.zeros((n, n), dtype=bool)
    for sigma in ab.sync:
        S |= _kron(gi.A[sigma], gj.A[ab.complement(sigma)])
    return S


def reachable(M: np.ndarray, start: int) -> np.ndarray:
    """Bool vector of states reachable from `start` (including start)."""
    seen = np.zeros(M.shape[0], dtype=bool)
    seen[start] = True
    frontier = seen.copy()
    while frontier.any():
        nxt = M[frontier].any(axis=0) & ~seen
        seen |= nxt
        frontier = nxt
    return seen


def pair_compatibility(gi: Graph, gj: Graph) -> int:
    """Number of product states reachable from (gi.x, gj.x) that have at
    least one outgoing sync edge (non-parity relation)."""
    start = gi.x * gj.A.shape[1] + gj.x
    M = product_relation(gi, gj)
    R = reachable(M, start)
    has_sync = _sync_relation(gi, gj).any(axis=1)
    return int((R & has_sync).sum())
