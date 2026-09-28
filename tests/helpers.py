"""Builders for synthetic traces, shared by tests across modules."""

from __future__ import annotations

from model.actions import Alphabet
from model.trace import ProbeEvent, Tick


def build_trace(
    moves: list[tuple[int, int | None, bool]],
    a: int = 0,
    b: int = 1,
    first: int = 0,
    events: dict[int, tuple[ProbeEvent, ...]] | None = None,
) -> list[Tick]:
    """Strictly alternating trace between agents a and b.

    moves[t] = (action, response, accepted); response None for τ.
    `first` is the id of the first initiator. Node positions are left at 0.
    """
    events = events or {}
    trace = []
    init, resp = (first, b if first == a else a)
    for t, (action, response, accepted) in enumerate(moves):
        trace.append(
            Tick(t, init, resp, action, response, accepted, 0, 0, events.get(t, ()))
        )
        init, resp = resp, init
    return trace


def accepted_offer(alphabet: Alphabet, sigma: int) -> tuple[int, int, bool]:
    return (sigma, alphabet.complement(sigma), True)


def declined_offer(alphabet: Alphabet, sigma: int) -> tuple[int, int, bool]:
    return (sigma, alphabet.tau, False)


def tau(alphabet: Alphabet) -> tuple[int, None, bool]:
    return (alphabet.tau, None, False)
